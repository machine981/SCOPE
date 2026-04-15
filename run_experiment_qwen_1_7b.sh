#!/bin/bash

# --- 模型 ---
POLICY_MODEL_PATH=./Models/Qwen3-1.7B-Base
TEACHER_MODEL_NAME=Qwen3-8B
IP_POOL="['xx.xxx.x.xx','...']"

# --- 数据 ---
TRAIN_DATA=./verl-distillation-ori/data/deepmath_new/deepmath_new_train.parquet
VAL_DATA=./verl-distillation-ori/data/aime/test.parquet

# --- 实验名称（决定 wandb 名称和模型保存路径）---
EXTRA_NAME=xxxxxxxxx

# --- 训练超参 ---
LR=5e-5
TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=256
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=12288
ROLLOUT_N=8                  # 每个 prompt 采样条数，OPD=1，GRPO=8
TEMPERATURE=0.6
GPU_MEMORY_UTILIZATION=0.75
TOTAL_EPOCHS=10
SAVE_FREQ=8
TEST_FREQ=-1

# --- 算法配置 ---
ADV_ESTIMATOR=none           # none（纯OPD）/ grpo（GRPO）
DISTILL_ENABLE=True          # 是否启用蒸馏
DISTILL_SIGNAL_IN_LOSS=False # 蒸馏信号是否进 loss
DISTILL_LOSS_TYPE=low_var_kl # kl / low_var_kl
KL_LOSS_TYPE=kl              # Teacher 端 KL 类型
USE_KL_LOSS=False            # 是否使用 KL loss
KL_LOSS_COEF=0.0001
KL_COEF=0.0001

USE_SCOPE_DUAL_PATH_WEIGHTING=True
SCOPE_TAU=1.0
SCOPE_USE_SEQ_WEIGHTS=True
USE_STUDENT_PATH_WEIGHTS=True
USE_TEACHER_PATH_WEIGHTS=True
STUDENT_PATH_PPL_POSITIVE=True
TEACHER_PATH_PPL_POSITIVE=True

# --- Wandb 项目名 ---
PROJECT_NAME=opd_distill_mixed

# 环境配置
export PYTHONPATH="$PYTHONPATH:./conda/envs/verl_opd_topk/bin"
export PATH="./conda/envs/verl_opd_topk/bin:$PATH"

PROJECT_PATH=./verl-distillation-ori
cd $PROJECT_PATH
CUR_DIR=$PROJECT_PATH

# 自动解析环境参数
NNODES=`python parse_env.py nnodes`
MASTER_ADDR=`python parse_env.py master_addr`
GPUS_PER_NODE=`python parse_env.py nproc_per_node`
NODE_RANK=`python parse_env.py node_rank`
WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))
PORT=6379

# 保存路径
SAVE_DIR=./Models/${EXTRA_NAME}

export RAY_memory_monitor_refresh_ms=0
export VLLM_USE_V1=1
export WANDB_MODE=offline
export WANDB_DIR=${SAVE_DIR}/wandb
mkdir -p $WANDB_DIR

python3 ./verl-distillation-ori/verl/utils/low_gpu_utilization_new.py --gpu_number $GPUS_PER_NODE &
echo "low_gpu_utilization.py started"

if [ "$NODE_RANK" == "0" ]; then
    echo "HEAD NODE"
    ray start --head --node-ip-address $MASTER_ADDR --port=${PORT} --num-gpus ${GPUS_PER_NODE}
    sleep 30

    ray job submit --runtime-env-json="{\"working_dir\": \"$CUR_DIR\",
                        \"env_vars\": {
                            \"PATH\": \"$PATH\",
                            \"GPUS_PER_NODE\": \"$GPUS_PER_NODE\"
                        },
                        \"excludes\": [
                            \"afo-base-0.0.1-SNAPSHOT.jar\",
                            \"jdk-11.0.12/lib/modules\",
                            \"jdk-11.0.12/lib/src.zip\",
                            \"jdk-11.0.12/lib/server/libjvm.so\",
                            \"jdk-11.0.12/jmods/java.base.jmod\",
                            \"jdk-11.0.12/jmods/java.desktop.jmod\",
                            \"data/*\",
                            \"hope.code.tar.gz\",
                            \"afo-dist-user-files.tar.gz\",
                            \"libbwfs76.so\",
                            \"apex/*\",
                            \"docker/*\",
                            \".git/*\",
                            \"temp_ray/*\"
                        ]}" \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${POLICY_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.distill_signal_in_loss=${DISTILL_SIGNAL_IN_LOSS} \
    actor_rollout_ref.actor.distill_loss_type=${DISTILL_LOSS_TYPE} \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    distill.enable=${DISTILL_ENABLE} \
    distill.model_name=${TEACHER_MODEL_NAME} \
    distill.ip_pool=${IP_POOL} \
    distill.kl_loss_type=${KL_LOSS_TYPE} \
    reward_model.enable=False \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    algorithm.use_scope_dual_path_weighting=${USE_SCOPE_DUAL_PATH_WEIGHTING} \
    algorithm.scope_tau=${SCOPE_TAU} \
    algorithm.scope_use_seq_weights=${SCOPE_USE_SEQ_WEIGHTS} \
    algorithm.teacher_path_ppl_positive=${TEACHER_PATH_PPL_POSITIVE} \
    algorithm.student_path_ppl_positive=${STUDENT_PATH_PPL_POSITIVE} \
    algorithm.use_student_path_weights=${USE_STUDENT_PATH_WEIGHTS} \
    algorithm.use_teacher_path_weights=${USE_TEACHER_PATH_WEIGHTS} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXTRA_NAME} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_DIR}

else
    echo "WORKER NODE"
    sleep 10
    ray start --block --address=${MASTER_ADDR}:${PORT} --num-gpus ${GPUS_PER_NODE}
fi
