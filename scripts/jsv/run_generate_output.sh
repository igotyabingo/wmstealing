#!/bin/bash 
 
echo "### START DATE=$(date)" 
echo "### HOSTNAME=$(hostname)" 
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" 
 

source  ~/.bashrc 
conda activate ws
 

ml unload cuda/11.2 nccl/2.8.4/cuda11.2 
ml load cuda/11.0 nccl/2.8.4/cuda11.0 
ml list 
 
# "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen3-4B-Instruct-2507" "meta-llama/Llama-3.2-1B-Instruct"
target_model_id='Qwen/Qwen3-4B-Instruct-2507'

max_new_token=400
# "c4_realnews" (base= T, F), "dolly_cw" (base= F), "mmw_book_report" (base= F)
prompt_set_name="c4_realnews"
delta_actual=4.0
base="False"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -o allexport
source .env
set +o allexport

huggingface-cli login --token $HUGGINGFACE_TOKEN
cd ./jsv
 
python generate_output.py \
    --target_model_id $target_model_id \
    --prompt_set_name $prompt_set_name \
    --max_new_token $max_new_token \
    --delta_actual $delta_actual \
    --base $base

echo "###" 
echo "### END DATE=$(date)"