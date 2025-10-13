import torch
from generation_dataset import GenerationPrompts
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import torch.nn.functional as F

from datasets import load_dataset
from utils import get_ngram_red_green_list
from extended_watermark_processor import WatermarkLogitsProcessor

import argparse
import tqdm
import json

def generate(
    target_model,
    tokenizer,
    raw_prompt,
    prompt_ids,
    max_new_token,
    delta_actual,
    base,
):
    attention_mask = torch.ones_like(prompt_ids).to(target_model.device)
    generation_kwargs = dict(
        max_new_tokens=max_new_token,
        do_sample=True,                
        pad_token_id=tokenizer.eos_token_id,  
        attention_mask=attention_mask,
    )

    if base == "False":
        vocab = tokenizer.get_vocab().values()
        processor = WatermarkLogitsProcessor(
            vocab=vocab,
            gamma=0.25,
            delta=4.0,
            seeding_scheme="selfhash",
            device=prompt_ids.device,
            tokenizer=tokenizer,
        )
        generation_kwargs["logits_processor"] = LogitsProcessorList([processor])

    output_ids = target_model.generate(
        prompt_ids.to(target_model.device),
        **generation_kwargs
    )

    result_dict = {}
    result_dict["prompt"] = tokenizer.decode(prompt_ids[0])
    result_dict["raw_prompt"] = raw_prompt
    result_dict["generated_text"] = tokenizer.decode(output_ids[0][len(prompt_ids[0]):])

    return result_dict


def main(args):
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    prompt_set_name = args.prompt_set_name # c4_realnews, mmw_book_report, dolly_cw

    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_id)
    print("model loading") # for debug
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_id,
        device_map="auto",
        torch_dtype="auto"
    )
    target_model.eval()
    print("model loaded") # for debug

    target_gen_dataset = GenerationPrompts(
        model_id=args.target_model_id,
        prompt_set_name=prompt_set_name,
        tokenizer=target_tokenizer,
    )
    
    generation_args = {
        "delta_actual": args.delta_actual,
        "max_new_token": args.max_new_token,
    }

    with torch.no_grad():
        for idx in range(len(target_gen_dataset)):
            if idx%100 == 0:
                torch.cuda.empty_cache()
            prompt_ids, raw_prompt, prompt_idx = target_gen_dataset[idx]
            print("running idx: ", prompt_idx) # for debug
            assert torch.equal(
                prompt_ids, target_gen_dataset[idx][0]
            )  # make sure the tokenized results are exactly the same

            result_dict = generate(
                target_model=target_model,
                tokenizer=target_tokenizer,
                raw_prompt=raw_prompt,
                prompt_ids=prompt_ids,
                max_new_token=args.max_new_token,
                delta_actual=args.delta_actual,
                base=args.base,
            )

            result_dict["generation_args"] = generation_args
            result_dict["prompt_idx"] = prompt_idx

            if args.base == "True":
                save_dir = f"./result/target_generation_jsv/{args.target_model_id}_base_{prompt_set_name}.jsonl"
            else:
                save_dir = f"./result/target_generation_jsv/{args.target_model_id}_wm_{prompt_set_name}.jsonl"
            with open(save_dir, "a",) as f:
                f.write(json.dumps(result_dict) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--prompt_set_name", type=str, default="c4_realnews")

    parser.add_argument("--max_new_token", type=int, default=800)
    parser.add_argument("--delta_actual", type=float, default=4)
    parser.add_argument("--base", type=str, default="False")

    args = parser.parse_args()

    main(args)
