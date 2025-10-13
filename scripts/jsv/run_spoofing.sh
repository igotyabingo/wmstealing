#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


# 충돌 방지를 위해 LD_LIBRARY_PATH 초기화
unset LD_LIBRARY_PATH
# Conda 환경 활성화
source ~/.bashrc
conda activate ws

# HPC CUDA 모듈은 제거 (Conda CUDA 런타임 사용)
module purge

# 필요하면 다른 모듈만 로드 (예: java)
module load java/22.0.2


# 작업 디렉토리 이동
cd ./jsv

# "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "Qwen/Qwen3-4B-Instruct-2507"
target_model_id='Qwen/Qwen3-4B-Instruct-2507'
# "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "Qwen/Qwen2.5-7B-Instruct"  
surrogate_model_id='Qwen/Qwen2.5-7B-Instruct'
# "mmw_book_report", "dolly_cw" "c4_test"
prompt_set_name="c4_test"
prevctx_width=3

# PyTorch GPU 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Python 스크립트 실행
python watermark_exploitation.py \
    --target_model_id $target_model_id \
    --surrogate_model_id $surrogate_model_id \
    --prompt_set_name $prompt_set_name \
    --prevctx_width $prevctx_width

echo "### END DATE=$(date)"