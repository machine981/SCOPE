#!/bin/bash

set -e

export PATH="./conda/envs/verl_opd_topk/bin:$PATH"
export PYTHONPATH="./verl-distillation-ori:$PYTHONPATH"
export PYTHONUNBUFFERED=1

PROJECT_PATH=./verl-distillation-ori
cd $PROJECT_PATH

# Data
DATA_PATH="$PROJECT_PATH/data/deepmath_new/deepmath_new_train.parquet"
SAMPLE_SIZE=2000
STUDENT_MODEL_PATH="./Models/DeepSeek-R1-Distill-Qwen-1.5B"

# Student generation params
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.6}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
BATCH_SIZE=${BATCH_SIZE:-1024}
N_SAMPLES=${N_SAMPLES:-4}

# Binning params
N_BINS=${N_BINS:-4}
SPLIT_METHOD=${SPLIT_METHOD:-"quartile"}

# Continuation params
N_CONTINUATIONS=${N_CONTINUATIONS:-4}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-0.6}
TRUNCATE_RATIO=${TRUNCATE_RATIO:-0.6}
MAX_TOKENS=${MAX_TOKENS:-32768}
FILTER_TRUNCATED=${FILTER_TRUNCATED:-0}

# Teacher model
TEACHER_IP_POOL="xx.xxx.xx.xx,..."
TEACHER_MODEL_NAME="Skywork-OR1-7B"
TEACHER_MAX_MODEL_LEN=34000

# Output dirs
TRUNCATE_RATIO_STR=$(echo "$TRUNCATE_RATIO" | tr '.' 'p')
GEN_DIR="$PROJECT_PATH/exp2_results/sample${SAMPLE_SIZE}_rollout${N_SAMPLES}"
PPL_DIR="$PROJECT_PATH/exp2_results/sample${SAMPLE_SIZE}_rollout${N_SAMPLES}_bins${N_BINS}"
OUTPUT_DIR="$PROJECT_PATH/exp2_results/sample${SAMPLE_SIZE}_rollout${N_SAMPLES}_bins${N_BINS}_trunc${TRUNCATE_RATIO_STR}"

# Skip flags
SKIP_STUDENT_GEN=${SKIP_STUDENT_GEN:-0}
SKIP_LABEL=${SKIP_LABEL:-0}
SKIP_PPL=${SKIP_PPL:-0}
SKIP_PREFIX=${SKIP_PREFIX:-0}

SAMPLE_PATH="$GEN_DIR/train_sample.parquet"
STUDENT_GEN_PATH="$GEN_DIR/student_gen.parquet"

mkdir -p "$GEN_DIR" "$PPL_DIR" "$OUTPUT_DIR"

if [ "$SKIP_STUDENT_GEN" != "1" ]; then
    # Sample training data
    python3 scripts/experiments/exp2_teacher_hallucination.py \
        --step prepare_data \
        --data_path "$DATA_PATH" \
        --sample_size $SAMPLE_SIZE \
        --output_dir "$GEN_DIR"

    # Parse environment for distributed training
    NNODES=`python3 parse_env.py nnodes`
    MASTER_ADDR=`python3 parse_env.py master_addr`
    GPUS_PER_NODE=`python3 parse_env.py nproc_per_node`
    NODE_RANK=`python3 parse_env.py node_rank`
    PORT=6379

    export RAY_memory_monitor_refresh_ms=0
    export VLLM_USE_V1=1

    if [ "$NODE_RANK" == "0" ]; then
        NUM_CPUS_FOR_RAY=$((GPUS_PER_NODE * 10 + 10))
        ray start --head --node-ip-address $MASTER_ADDR --port=${PORT} --num-gpus ${GPUS_PER_NODE} --num-cpus ${NUM_CPUS_FOR_RAY}
        sleep 10

        ray job submit \
            --runtime-env-json="{
                \"working_dir\": \"$PROJECT_PATH\",
                \"env_vars\": {
                    \"PATH\": \"$PATH\",
                    \"PYTHONPATH\": \"$PROJECT_PATH\",
                    \"VLLM_USE_V1\": \"1\"
                },
                \"excludes\": [
                    \"data/*\", \".git/*\", \"temp_ray/*\", \"exp2_results/*\"
                ]}" \
            -- python3 -m verl.trainer.main_generation \
                trainer.nnodes=$NNODES \
                trainer.n_gpus_per_node=$GPUS_PER_NODE \
                data.path="$SAMPLE_PATH" \
                data.prompt_key=prompt \
                data.n_samples=$N_SAMPLES \
                data.output_path="$STUDENT_GEN_PATH" \
                data.batch_size=$BATCH_SIZE \
                model.path="$STUDENT_MODEL_PATH" \
                rollout.temperature=$ROLLOUT_TEMPERATURE \
                rollout.top_k=20 \
                rollout.top_p=0.95 \
                rollout.prompt_length=$MAX_PROMPT_LENGTH \
                rollout.response_length=$MAX_RESPONSE_LENGTH \
                rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
                rollout.tensor_model_parallel_size=1 \
                rollout.enforce_eager=False \
                rollout.free_cache_engine=False \
                rollout.enable_chunked_prefill=False

        ray stop
    else
        ray start --block --address=${MASTER_ADDR}:${PORT} --num-gpus ${GPUS_PER_NODE}
    fi
fi

# Build skip args
SKIP_ARGS=""
[ "$SKIP_LABEL" == "1" ] && SKIP_ARGS="$SKIP_ARGS --skip_label"
[ "$SKIP_PPL" == "1" ] && SKIP_ARGS="$SKIP_ARGS --skip_ppl"
[ "$SKIP_PREFIX" == "1" ] && SKIP_ARGS="$SKIP_ARGS --skip_prefix"
[ "$FILTER_TRUNCATED" == "1" ] && SKIP_ARGS="$SKIP_ARGS --filter_truncated --max_response_length $MAX_RESPONSE_LENGTH"

# Teacher analysis: label + PPL + split + continuation + stats
python3 scripts/experiments/exp2_teacher_hallucination.py \
    --step analyze \
    --student_gen_path "$STUDENT_GEN_PATH" \
    --student_model_path "$STUDENT_MODEL_PATH" \
    --teacher_ip_pool "$TEACHER_IP_POOL" \
    --teacher_model_name "$TEACHER_MODEL_NAME" \
    --ppl_dir "$PPL_DIR" \
    --n_bins $N_BINS \
    --split_method "$SPLIT_METHOD" \
    --truncate_ratio $TRUNCATE_RATIO \
    --teacher_max_model_len $TEACHER_MAX_MODEL_LEN \
    --n_continuations $N_CONTINUATIONS \
    --teacher_temperature $TEACHER_TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --output_dir "$OUTPUT_DIR" \
    $SKIP_ARGS
