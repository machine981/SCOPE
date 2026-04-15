#!/bin/bash
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

INFER_PATH=./verl-distillation-ori/scripts/infer
cd $INFER_PATH

export PYTHONPATH="$PYTHONPATH:./conda/envs/vllm_qwen3/lib/python3.10/site-packages"
export PATH="./conda/envs/vllm_qwen3/bin:$PATH"

echo $PATH

debug=1

max_new_tokens=32768
temperature=0.6
top_p=0.95
top_k=20

# 检查参数数量
if [ $# -ne 10 ]; then
    echo "Usage: $0 <model_name_or_path> <json_file> <output_path> <rollout> <max_new_tokens> <max_model_len> <temperature> <top_p> <top_k> <repetition_penalty>"
    exit 1
fi

model_name_or_path=$1
json_file=$2
output_path=$3
rollout=$4
max_new_tokens=$5
max_model_len=$6
temperature=$7
top_p=$8
top_k=$9
repetition_penalty=${10}

PYTHON="python"
export CMD=" \
$PYTHON $INFER_PATH/general_inference.py \
    --json_file  $json_file \
    --output_path $output_path \
    --model_name_or_path $model_name_or_path \
    --max_new_tokens $max_new_tokens \
    --max_model_len $max_model_len \
    --temperature $temperature \
    --rollout $rollout \
    --top_p $top_p \
    --top_k $top_k \
    --repetition_penalty $repetition_penalty
"

set -e

echo $CMD

$CMD

wait