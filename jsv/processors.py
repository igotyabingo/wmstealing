import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    LogitsProcessor,
    TemperatureLogitsWarper,
)

from count_store import CountStore

class SpoofedProcessor(LogitsProcessor):
    def __init__(
        self,
        *args: Any,
        counts_base: CountStore,
        counts_wm: CountStore,
        prevctx_width: int,
        vocab_size: int,
        tokenizer: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.counts_base = counts_base
        self.counts_wm = counts_wm
        self.prevctx_width = prevctx_width
        self.vocab_size = vocab_size  # unused, see note below

        self.spoofer_strength = 8.25 # set 8.25 for spoofing attack, -10 for scrubbing attack
        self.min_wm_mass_empty = 0.00007
        self.min_wm_count_nonempty = 2
        self.clip_at = 2.0
        self.w_abcd = 2.0
        self.w_partials = 1.0
        self.w_empty = 0.5
        self.use_sampling = True
        self.sampling_temp = 0.7

        self.tokenizer = tokenizer
        self.emptyctx = tuple([-1 for _ in range(self.prevctx_width)])
    
        # Cache for boosts
        self.boosts_cache: Dict[str, torch.Tensor] = {}

        # Estimates
        self.green_estimates: List[List[float]] = []
        self.z_estimates: List[List[float]] = []

    def _compute_future_token_ppl(
        self,
        pre_ids: torch.Tensor,
        post_ids: torch.Tensor,
        device: str,
        num_options_per_token: int = 10,
        num_tokens: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of pre_ids and post_ids, compute the perplexity of the top num_options_per_token tokens for each masked token.
        Due to tokenization errors some may be masked away (probs=0) so it's not necessarily top num_options_per_token best choices;
        for similar reason they may not be sorted by probability, so use with care.

        Args:
            pre_ids: (batch_size, any_length)   # Generally assumed to be of equal length
            post_ids: (batch_size, any_length)  # Generally assumed to be of length 1 - The green token
            num_options_per_token: int          # Number of options to return for each masked token
            num_tokens: int                     # Number of masked tokens to consider

        Returns:
            resulting_tokens: (batch_size, num_tokens, num_options_per_token)    # The token ids of the top num_options_per_token tokens for each masked token
            resulting_probs: (batch_size, num_tokens, num_options_per_token)     # The probabilities of the top num_options_per_token tokens for each masked token
        """
        self.ppl_model.to(device)

        b = pre_ids.shape[0]

        pre_text = self.tokenizer.batch_decode(pre_ids, skip_special_tokens=True)
        post_text = self.tokenizer.batch_decode(post_ids, skip_special_tokens=True)

        input_txt = [x + "<mask>" * num_tokens + y for x, y in zip(pre_text, post_text)]

        inputs = self.ppl_tokenizer(input_txt, return_tensors="pt", padding=True).to(device)
        predictions = self.ppl_model(**inputs)[0]

        # Find the positions of the <mask> tokens
        mask = (inputs.input_ids == self.ppl_tokenizer.mask_token_id).unsqueeze(-1)

        selected_tokens = torch.masked_select(predictions, mask).reshape((b, num_tokens, -1))
        st_probs = torch.nn.functional.softmax(selected_tokens, dim=-1)
        # Format [batch_size, num_masked_tokens, logits]

        top_probs, top_tokens = torch.topk(st_probs, k=num_options_per_token, sorted=True, dim=-1)
        # Format [batch_size, num_masked_tokens, num_options_per_token]

        # Decode the tokens
        top_strings = self.ppl_tokenizer.batch_decode(
            top_tokens.reshape((-1, 1)), skip_special_tokens=True
        )
        assert len(top_strings) == b * num_tokens * num_options_per_token
        top_strings = [
            top_strings[i * num_options_per_token : (i + 1) * num_options_per_token]
            for i in range(b * num_tokens)
        ]

        # Re-tokenize with the other tokenizer
        top_tokenized = [
            self.tokenizer(x, return_tensors="pt", padding=True, add_special_tokens=False)
            for x in top_strings
        ]

        resulting_tokens = torch.zeros_like(top_probs, dtype=torch.long, device=device)
        resulting_probs = torch.zeros_like(top_probs, dtype=torch.float, device=device)

        spec_ids = torch.tensor(self.spec_ids, dtype=torch.long, device=device)
        for i in range(b):
            for j in range(num_tokens):
                comb_index = i * num_tokens + j
                curr_tokens = top_tokenized[comb_index].input_ids.to(device)  # [options, 1?]

                # Kick out (set prob=0) those where
                # i) tokenizer didn't produce a single token
                # ii) the candidate is a special token
                good_mask = torch.ones((num_options_per_token,), dtype=torch.bool, device=device)
                if curr_tokens.shape[1] > 1:
                    expected_nb_pad = curr_tokens.shape[1] - 1
                    nb_pad = torch.sum(curr_tokens == self.tokenizer.pad_token_id, dim=-1)
                    good_mask &= nb_pad == expected_nb_pad
                good_mask &= ~torch.isin(curr_tokens[:, -1], spec_ids)

                # Extract and mask out bad ones
                resulting_tokens[i, j, :] = curr_tokens[:, -1]
                resulting_probs[i, j, :] = top_probs[i, j, :]
                resulting_probs[i, j, ~good_mask] = 0.0

        return resulting_tokens, resulting_probs

    # Given an ordered (tok|-1){3} or unordered (tok){0,3} context return a combined and
    # normalized boost vector with boosts from 0 to 1 for each token in the vocab
    def get_boosts(
        self, ctx: tuple, vocab_sz: int, ordered: bool, device: str, normalize: bool = True
    ) -> torch.Tensor:
        # Get counts and mass for base and watermarked
        counts_base: Dict[int, int] = self.counts_base.get(ctx, ordered)
        counts_base_tensor = torch.sparse_coo_tensor(
            torch.tensor([list(counts_base.keys())]),
            list(counts_base.values()),
            [vocab_sz],
            device=device,
        ).to_dense()
        mass_base = counts_base_tensor / (counts_base_tensor.sum() + 1e-6)

        counts_wm: Dict[int, int] = self.counts_wm.get(ctx, ordered)
        counts_wm_tensor = torch.sparse_coo_tensor(
            torch.tensor([list(counts_wm.keys())]),
            list(counts_wm.values()),
            [vocab_sz],
            device=device,
        ).to_dense()
        mass_wm = counts_wm_tensor / (counts_wm_tensor.sum() + 1e-6)


        # Compute ratios, small/0 is bad, large/0 is very good
        if (ordered and ctx == self.emptyctx) or (not ordered and len(ctx) == 0):
            min_data_thresh = round(self.min_wm_mass_empty * counts_base_tensor.sum().item())
        else:
            min_data_thresh = self.min_wm_count_nonempty
        enough_data_mask = counts_wm_tensor >= min_data_thresh
        base_zero_mask = counts_base_tensor == 0

        # Get mass ratios, handles nonexistence nicely
        ratios = torch.zeros_like(mass_wm)  # stays 0 where not enough data
        core_mask = enough_data_mask & ~base_zero_mask
        ratios[core_mask] = mass_wm[core_mask] / mass_base[core_mask]
        ratios[enough_data_mask & base_zero_mask] = max(1, ratios.max().item()) + 1e-3

        if normalize:
            # Compute boost values by clipping
            ratios[ratios < 1] = 0
            ratios[ratios > self.clip_at] = self.clip_at
            ratios /= self.clip_at  # now 0 or 0.5-1
            # Tie break by wm counts (important for future)
            if ratios.max() > 0:
                ratios[ratios > 0] += (
                    counts_wm_tensor[ratios > 0] / counts_wm_tensor[ratios > 0].max()
                ) * 1e-4
                ratios = ratios / ratios.max()  # Should still be [0,1]

        return ratios

    # Optimization: if needed boosts are cached skip the call to .get_boosts()
    # Esp. useful for empty tuple that never changes (but useful also for pairings)
    def _get_boosts_with_cache(
        self, ctx: tuple, vocab_sz: int, ordered: bool, device: str
    ) -> torch.Tensor:
        k = str((ctx, ordered))
        if k not in self.boosts_cache:
            self.boosts_cache[k] = self.get_boosts(ctx, vocab_sz, ordered, device)
        return self.boosts_cache[k]

    def __call__(  # noqa: C901
        self, input_ids: torch.LongTensor, logits: torch.Tensor
    ) -> torch.Tensor:
        # Check if there is enough input_ids to apply the processor
        if input_ids.shape[1] < self.prevctx_width:
            return logits

        # input_ids: (B, maxlen) | logits: (B, vocab_size)
        # NOTE: this can be >self.vocab_size (32128 vs 32100 for T5, last 28 ignored)
        device = str(logits.device)
        vocab_sz = logits.shape[1]

        for b in range(input_ids.shape[0]):
            # Boosts per token are a weighted sum of several contribs
            boosts = torch.zeros((vocab_sz,), device=device)
            total_w = 0.0
            ctx: Tuple[Optional[int], ...] = tuple()  # gptwm
            if self.prevctx_width > 0:
                ctx = tuple(input_ids[b][-self.prevctx_width :].cpu().tolist())

            # 1) {ABC}->D: most precise but generally sparse
            boosts_abcd = self._get_boosts_with_cache(tuple(sorted(ctx)), vocab_sz, False, device)
            boosts += self.w_abcd * boosts_abcd
            total_w += self.w_abcd

            if self.w_partials > 0:
                # 2) Find S, the strongest among {ABC} by looking at {AB}->D and then add {S}->D
                # (so either D is weak and (S,D) is good or D is strong and (D,D) is good) -> both ok
                solo_boosts = []
                for tok in ctx:
                    solo_boosts.append(self._get_boosts_with_cache((tok,), vocab_sz, False, device))
                pair_boosts: List[List[torch.Tensor]] = [
                    [torch.tensor([]) for _ in range(len(ctx))] for _ in range(len(ctx))
                ]
                for i, toki in enumerate(ctx):
                    for delta, tokj in enumerate(ctx[i + 1 :]):
                        j = i + 1 + delta
                        pair_ctx = tuple(sorted([toki, tokj]))  # type: ignore
                        pair_boosts[i][j] = self._get_boosts_with_cache(
                            pair_ctx, vocab_sz, False, device
                        )
                        pair_boosts[j][i] = pair_boosts[i][j]

                # To be strong you need to have higher cossim with pair boost than all
                winner = -1
                cossim = F.cosine_similarity
                for i, toki in enumerate(ctx):
                    is_strong = True
                    for j, tokj in enumerate(ctx):
                        if i == j:
                            continue
                        cossim_i = cossim(pair_boosts[i][j], solo_boosts[i], dim=0).item()
                        cossim_j = cossim(pair_boosts[i][j], solo_boosts[j], dim=0).item()
                        is_strong &= cossim_i > cossim_j
                    if is_strong:
                        if winner == -1:
                            winner = i
                        else:
                            winner = -2
                            break

                # If there was a unique winner add its boost
                if winner > -1:
                    boosts += self.w_partials * solo_boosts[winner]
                    total_w += self.w_partials

            if self.w_empty > 0:
                # 3) Just use the empty context {}->D for an additional ctx-independent boost
                # (finding D that are strong + true)
                boosts_empty = self._get_boosts_with_cache(tuple(), vocab_sz, False, device)
                boosts += self.w_empty * boosts_empty
                total_w += self.w_empty

            # Average to get the final boosts
            boosts /= total_w
            new_logits = logits[b] + self.spoofer_strength * boosts

            # New Z-score estimation
            if len(self.z_estimates) < b + 1:
                self.green_estimates.append([])
                self.z_estimates.append([])
            green_probs = boosts * 0.75 + 0.25
            green_probs[green_probs > 1.0] = 1.0

            if self.use_sampling:
                warper = TemperatureLogitsWarper(self.sampling_temp)
                sampling_probs = warper(input_ids[b], logits[b]).softmax(0)
            else:
                sampling_probs = torch.zeros_like(logits[b], device=logits.device)
                sampling_probs[logits[b].argmax()] = 1.0

            total_green_prob = (green_probs * sampling_probs).sum().cpu().item()
            self.green_estimates[b].append(total_green_prob)
            N = sum(self.green_estimates[b])
            T = len(self.green_estimates[b])
            gamma = 0.25  # TODO make a param if running on different gammas
            self.z_estimates[b].append(((N - gamma * T) / math.sqrt(gamma * (1 - gamma) * T)))

            # Finally apply the boosts
            logits[b] = new_logits
        return logits