#!/bin/bash 

source  ~/.bashrc 
conda activate ws

set -o allexport
source .env
set +o allexport

openai_api_key=$OPENAI_KEY
# "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen3-4B-Instruct-2507"
target_model_id="Qwen/Qwen3-4B-Instruct-2507" 
# "dolly_cw", "mmw_book_report" "c4_test"
prompt_set_name="dolly_cw"
cd ./demark
blackbox=1

python evaluate_results.py \
    --target_model_id $target_model_id \
    --prompt_set_name $prompt_set_name \
    --openai_api_key $openai_api_key  \
    --blackbox $blackbox