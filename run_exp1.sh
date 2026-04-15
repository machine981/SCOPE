#!/bin/bash

set -e

export PATH="./conda/envs/verl_opd_topk/bin:$PATH"
export PYTHONPATH="./verl-distillation-ori:$PYTHONPATH"
export PYTHONUNBUFFERED=1

PROJECT_PATH=./verl-distillation-ori
cd $PROJECT_PATH

# Wrong samples from exp2
WRONG_SAMPLES_PATH="$PROJECT_PATH/exp2_results/sample2000_rollout4_bins4/wrong_samples_with_ppl.parquet"
# Teacher model
TEACHER_IP_POOL="xx.xxx.xx.xx,..."
TEACHER_MODEL_NAME="Skywork-OR1-7B"
# Generation params
N_SAMPLES=${N_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-0.6}
MAX_TOKENS=${MAX_TOKENS:-32768}
# Output
OUTPUT_DIR="$PROJECT_PATH/exp1_results"

echo "Exp1: Teacher avg@${N_SAMPLES}"

python3 scripts/experiments/exp1_teacher_avg4.py \
    --wrong_samples_path "$WRONG_SAMPLES_PATH" \
    --teacher_ip_pool "$TEACHER_IP_POOL" \
    --teacher_model_name "$TEACHER_MODEL_NAME" \
    --n_samples $N_SAMPLES \
    --temperature $TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --output_dir "$OUTPUT_DIR"
