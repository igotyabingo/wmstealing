import argparse
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, LogitsProcessorList
import torch
import tqdm
import json
import torch.nn.functional as F

from processors import SpoofedProcessor
from count_store import CountStore
from generation_dataset import GenerationPrompts
from extended_watermark_processor import WatermarkDetector

import nltk
from nltk.tokenize import sent_tokenize

def load_queries_and_learn(tokenizer: AutoTokenizer, texts_wm: List[str], dest_counts: CountStore, prevctx_width: int):
    # load queries then update count_store with base or wm (learn)
    for text in texts_wm:
        toks = tokenizer(text)["input_ids"]  
        for i in range(prevctx_width, len(toks)):
            ctx = tuple(toks[i - prevctx_width : i])
            dest_counts.add(ctx, toks[i], 1)

def generate_dipper(target_model, tokenizer, prompt, spoofed_processor: SpoofedProcessor):    
    device = target_model.device

    lex_code = 40
    order_code = 80
    sent_interval = 3

    prefix, input_text = [tok.strip() for tok in prompt.split("|||")]
    input_text = " ".join(input_text.split())
    sentences = nltk.sent_tokenize(input_text)
    prefix = " ".join(prefix.replace("\n", " ").split())
    output_text = ""
    for sent_idx in range(0, len(sentences), sent_interval):
        curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
        final_input_text = f"lexical = {lex_code}, order = {order_code}"
        if prefix:
            final_input_text += f" {prefix}"
        final_input_text += f" <sent> {curr_sent_window} </sent>"
        input_ids = tokenizer(final_input_text, return_tensors="pt").input_ids.to(device)
        output_ids = target_model.generate(
            input_ids,
            logits_processor=LogitsProcessorList([spoofed_processor]),
            max_new_tokens = 800,
            use_cache=False,
            do_sample = True, 
            num_beams = 1, 
            temperature = 0.7, 
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prefix += " " + decoded_output
        output_text += " " + decoded_output
    
    return output_text
        

def run_eval_dipper(target_model, tokenizer, prompts, generations, save_dir, spoofed_processor, detector):
    target_gen_dataset = [f"{a} ||| {b}" for a, b in zip(prompts, generations)]

    with torch.no_grad(): 
        for idx in range(len(target_gen_dataset)):
            print("running idx: ", idx)
            
            if idx%3 == 0:
                torch.cuda.empty_cache()

            prompt = target_gen_dataset[idx]
            generated_text = generate_dipper(target_model, tokenizer, prompt, spoofed_processor=spoofed_processor) 

            p_value = detector.detect(generated_text) 

            result_dict = {}
            result_dict["raw_prompt"] = prompt
            result_dict["generated_text"] = generated_text
            result_dict["p_value"] = p_value
            result_dict["prompt_idx"] = idx

            with open(save_dir, "a",) as f:
                f.write(json.dumps(result_dict) + "\n")

def main(args):
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    counts_base = CountStore(args.prevctx_width)
    counts_wm = CountStore(args.prevctx_width)
    
    print("DIPPER loading\n")
    # surrogate_model = dipper
    surrogate_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
    # Used to be T5Tokenizer but we need this for offset_mapping
    surrogate_model = T5ForConditionalGeneration.from_pretrained(
        "kalpeshk2011/dipper-paraphraser-xxl",
        device_map="auto",
        torch_dtype="auto")
    surrogate_model.eval()
    print("DIPPER loaded")

    print("CountStore updating 1")
    base_text = []    
    with open(f"./result/target_generation_jsv/{args.target_model_id}_base_c4_realnews.jsonl", "r") as f:
        for line in f:  
            base_text.append(json.loads(line)["generated_text"]) 
    # get prompts and texts_wm (base) from proper file directory
    load_queries_and_learn(surrogate_tokenizer, base_text, counts_base, args.prevctx_width)
    
    print("CountStore updating 2")
    wm_text = []
    with open(f"./result/target_generation_jsv/{args.target_model_id}_wm_c4_realnews.jsonl", "r") as f:
        for line in f:  
            wm_text.append(json.loads(line)["generated_text"])
    load_queries_and_learn(surrogate_tokenizer, wm_text, counts_wm, args.prevctx_width)
    # CountStore updated

    prompt_set_name = args.prompt_set_name # mww or dolly

    # get evaluation dataset
    prompts = []
    generations = []
    with open(f"./result/target_generation_jsv/{args.target_model_id}_wm_{prompt_set_name}.jsonl", "r") as f:
        for line in f: 
            data = json.loads(line)
            prompts.append(data["raw_prompt"])
            generations.append(data["generated_text"])

    save_dir = f"./result/scrubbing/jsv/{prompt_set_name}/{args.target_model_id}_wm_{prompt_set_name}.jsonl"

    spoofed_processor = SpoofedProcessor(counts_base=counts_base, counts_wm=counts_wm, prevctx_width=args.prevctx_width, vocab_size=surrogate_tokenizer.get_vocab(), tokenizer=surrogate_tokenizer)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_id)
    detector = WatermarkDetector(vocab=target_tokenizer.get_vocab(), seeding_scheme='selfhash', gamma=0.25, device='cuda', tokenizer=target_tokenizer, normalizers=[], z_threshold=4.0, ignore_repeated_ngrams=True) 

    run_eval_dipper(surrogate_model, surrogate_tokenizer, prompts, generations, save_dir, spoofed_processor, detector)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_model_id", type=str, default="meta-llama/Llama-3.2-3B-Instruct" 
    )
    parser.add_argument("--prompt_set_name", type=str, default="mmw_book_report") 

    parser.add_argument("--prevctx_width", type=int, default=3)

    args = parser.parse_args()

    main(args)
