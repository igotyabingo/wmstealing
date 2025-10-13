#!/bin/bash 
 
echo "### START DATE=$(date)" 
echo "### HOSTNAME=$(hostname)" 
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" 
 

source  ~/.bashrc 
conda activate ws
 

ml unload cuda/11.2 nccl/2.8.4/cuda11.2 
ml load cuda/11.0 nccl/2.8.4/cuda11.0 
ml list 

gpu_num=1

# "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen3-4B-Instruct-2507"
model_id='meta-llama/Llama-3.2-3B-Instruct' 
delta_actual=2
delta_predicted=2
delta_reduce_multiplier=1 #eta


max_new_token=300
total_word_num=20
# "dolly_cw" "mmw_book_report" 
prompt_set_name="dolly_cw"

save_dir="./result/scrubbing/demark"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -o allexport
source .env
set +o allexport

huggingface-cli login --token $HUGGINGFACE_TOKEN

cd ./demark

task_num=1
task_idx=0

python exp_watermark_removal.py \
        --task_idx $task_idx \
        --total_task_num $task_num \
        --model_id $model_id \
        --prompt_set_name $prompt_set_name \
        --delta_actual $delta_actual \
        --delta_predicted $delta_predicted \
        --max_new_token $max_new_token \
        --total_word_num $total_word_num \
        --delta_reduce_multiplier $delta_reduce_multiplier \
        --wm_type "KGW" \
        --save_dir $save_dir 

echo "###" 
echo "### END DATE=$(date)"