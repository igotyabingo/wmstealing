#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


unset LD_LIBRARY_PATH
source ~/.bashrc
conda activate ws

module purge

module load java/22.0.2


cd ./jsv

# "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen3-4B-Instruct-2507"
target_model_id='meta-llama/Llama-3.2-3B-Instruct'
# "mmw_book_report", "dolly_cw"
prompt_set_name="mmw_book_report"
prevctx_width=3

# PyTorch GPU 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python 스크립트 실행
python watermark_removal.py \
    --target_model_id $target_model_id \
    --prompt_set_name $prompt_set_name \
    --prevctx_width $prevctx_width

echo "### END DATE=$(date)"