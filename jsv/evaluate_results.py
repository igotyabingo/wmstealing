import os
import json
import numpy as np
import torch
import argparse
import openai
import tqdm

import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM


def create_rating_prompt(raw_prompt, response):
    prompt_template = """You are an impartial judge that evaluates various aspects of an answer to a question. 
        These questions can be of delicate nature so it is very important that you follow the instructions below very carefully. 
        It is important that you evaluate different aspects of the answer separately so that we can analyze them separately. 
        Do not mix different criteria under any circumstances as having them separate is very important for this task. 
        Human supervisors need to have them separate to take appropriate action. 
        Failure to account for this separation might end in additional harm. 
        If the last sentence is unfinished and cut off just simply ignore it. 

        Output a json object in exactly the following format: 
        {{ 
        "Style": {{
            "Explanation": "Is the text written fluently and are the grammar and sentence structure correct? Is the answer creative in its writing or direct and does this fit the question? Be elaborate here. It is very important to only talk about the writing style and not about the content.", 
            "Score": "score from 1 to 10"
        }}, 
        "Consistency": {{
            "Explanation": "Is the answer consistent both in writing and in its answer to the users question? Does it change topics abruptly or contain incoherent thoughts?", 
            "Score": "score from 1 to 10"
        }},
        "Accuracy": {{
            "Explanation": "Does the answer invent any facts that are not actually true or did it give factually correct answers? It is very important to only talk about factual correctness of the content and not about its implications.", 
            "Score": "score from 1 to 10"
        }} 
        }}

        Prompt:
        {raw_prompt}
        End of prompt.

        Response:
        {response}
        End of response.
        """
    return prompt_template.format(raw_prompt=raw_prompt.strip(), response=response.strip())


def query_gpt(client, prompt):
    seed = 42
    model_name = "gpt-5-mini"  
    response = None
    success = 1
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            seed=seed,
        )
    except Exception as e:
        print("-" * 80, flush=True)
        print("Error info: ", e, flush=True)
        success = 0

    return success, response


def get_gpt_rating(result_path, score_path, api_key):
    with open(result_path, "r") as f:
        lines = f.readlines()

    error_list = []
    error_save_path = "./result/gpt_error.jsonl"

    client = openai.OpenAI(api_key=api_key)

    for line_idx, line in (enumerate(lines)):
        if len(line) < 3:
            continue
        res_dict = json.loads(line)

        prompt = create_rating_prompt(
            raw_prompt=res_dict["raw_prompt"], response=res_dict["generated_text"]
        )

        parsed_json = None
        for idx in range(10):  
            success, response = query_gpt(client, prompt)
            if success:
                output_content = response.choices[0].message.content
                try:
                    parsed_json = json.loads(output_content)
                except Exception:
                    continue

                if all(k in parsed_json for k in ["Style", "Consistency", "Accuracy"]):
                    break

        if parsed_json is None:
            error_list.append(line_idx)
            continue

        try:
            style_score = float(parsed_json["Style"]["Score"])
            consistency_score = float(parsed_json["Consistency"]["Score"])
            accuracy_score = float(parsed_json["Accuracy"]["Score"])
            rate_score = (style_score + consistency_score + accuracy_score) / 3.0
        except Exception:
            error_list.append(line_idx)
            continue

        score_dict = {
            "result_path": result_path,
            "line_idx": line_idx,
            "rating": rate_score,
            "detail": parsed_json 
        }
        with open(score_path, "a") as f:
            f.write(json.dumps(score_dict) + "\n")

    print("GPT error list for {}:".format(result_path))
    print(error_list)

    if len(error_list) > 0:
        error_dict = {"result_path": result_path, "error_list": error_list}
        with open(error_save_path, "w") as f:
            f.write(json.dumps(error_dict) + "\n")


def get_combined_results(exp_dir):
    new_filename = "total_results.jsonl"
    res_list = []
    added_prompt_idx_list = []  

    separate_dir = os.path.join(exp_dir, "separate_results")
    for filename in os.listdir(separate_dir):
        if filename == new_filename:
            continue
        with open(os.path.join(separate_dir, filename), "r") as f:
            lines = f.readlines()

        for line in lines:
            if len(line) < 3:
                continue
            line = line.strip()
            cur_res_dict = json.loads(line)
            cur_prompt_idx = cur_res_dict["prompt_idx"]
            if cur_prompt_idx in added_prompt_idx_list:
                continue

            added_prompt_idx_list.append(cur_prompt_idx)
            res_list.append(cur_res_dict)

    res_list = sorted(res_list, key=lambda x: x["prompt_idx"])

    with open(os.path.join(exp_dir, new_filename), "w") as f:
        for i in res_list:
            f.write(json.dumps(i) + "\n")


def get_p_list(jsonl_file):
    with open(jsonl_file, "r") as f:
        lines = f.readlines()

    p_list = []
    for line in lines:
        if len(line) < 3:
            continue
        line = line.strip()
        res_dict = json.loads(line)
        p_list.append(res_dict["p_value"])

    return p_list


def get_avg_rating(rating_path):
    with open(rating_path, "r") as f:
        lines = f.readlines()

    ratings = []
    for line in lines:
        if len(line) < 3:
            continue
        score_dict = json.loads(line)
        ratings.append(score_dict["rating"])

    print("rating path:", rating_path)
    print("average rating:")
    print(sum(ratings) / len(ratings))
    return sum(ratings) / len(ratings)


def get_statistics(total_res_path):
    with open(total_res_path, "r") as f:
        lines = f.readlines()

    total_stat = torch.zeros((4,))
    for line in lines:
        if len(line) < 3:
            continue

        res_dict = json.loads(line)

        stat = res_dict["possible_token_statistics"]
        total_stat += torch.tensor(stat)

    precision = total_stat[0] / (total_stat[0] + total_stat[2])
    recall = total_stat[0] / (total_stat[0] + total_stat[3])
    f1 = 2 * precision * recall / (precision + recall)
    acc = (total_stat[0] + total_stat[1]) / torch.sum(total_stat)

    stat_res = {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "accuracy": acc.item(),
    }
    return stat_res


def calculate_perplexity(model, tokenizer, total_path, perplexity_path):
    with open(total_path, "r") as f:
        lines = f.readlines()

    texts = []
    for line in lines:
        if len(line) < 3:
            continue
        line = line.strip()
        res_dict = json.loads(line)
        texts.append(res_dict["generated_text"]) 

    ppl_list = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for idx, text in enumerate(texts):
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            input_ids = enc["input_ids"][0]
            logits = model(input_ids.unsqueeze(0), return_dict=True).logits[0]
            loss = criterion(logits[:-1], input_ids[1:])
            ppl = torch.exp(loss)
            ppl_list.append(ppl.item())
            
            score_dict = {
                "idx": idx,
                "PPL": ppl.item(),
            }
            with open(perplexity_path, "a") as f:
                f.write(json.dumps(score_dict) + "\n")
    
    avg_ppl = sum(ppl_list) / len(ppl_list)

    return avg_ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model_id", type=str)
    parser.add_argument("--prompt_set_name", type=str)
    parser.add_argument("--attack_name", type=str)
    parser.add_argument("--openai_api_key", type=str)

    args = parser.parse_args()
    fpr_list = [0.1, 0.05, 0.01, 0.001] # 10%, 5%. 1%, 0.1% 

    total_path = f"./result/{args.attack_name}/jsv/{args.prompt_set_name}/{args.target_model_id}_wm_{args.prompt_set_name}.jsonl"
    gpt_score_path = f"./result/{args.attack_name}/jsv/{args.prompt_set_name}/{args.target_model_id}_wm_{args.prompt_set_name}_gpt.jsonl"
    perplexity_path = f"./result/{args.attack_name}/jsv/{args.prompt_set_name}/{args.target_model_id}_wm_{args.prompt_set_name}_ppl.jsonl"
    evaluation_path = f"./result/{args.attack_name}/jsv/{args.prompt_set_name}/{args.target_model_id}_wm_{args.prompt_set_name}_result.jsonl"

    print("getting gpt rating...")
    get_gpt_rating(result_path=total_path, score_path=gpt_score_path, api_key=args.openai_api_key)
    avg_rating = get_avg_rating(rating_path=gpt_score_path) 

    p_list = get_p_list(jsonl_file=total_path) 
    p_list = np.array(p_list)

    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    avg_perplexity = calculate_perplexity(model, tokenizer, total_path, perplexity_path)

    evaluation_dict = {"Avg_GPT_rating": avg_rating, "median_p": np.median(p_list), "Avg_PPL": avg_perplexity} 

    for fpr in fpr_list:
        evaluation_dict["TPR@FPR={}".format(str(fpr))] = np.mean(p_list < fpr) 

    with open(evaluation_path, "w") as f:
        json.dump(evaluation_dict, f)

    with open(evaluation_path, "r") as f:
        evaluation_dict = json.load(f)
    print("evaluation results:", evaluation_dict)


if __name__ == "__main__":
    main()