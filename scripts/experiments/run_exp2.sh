#!/bin/bash
# 实验二：证明"盲目蒸馏会传递教师的幻觉（Confidently Wrong）"
#
# 两阶段流程：
#   阶段一（需要 GPU）：用 main_generation.py（Ray + FSDP + vLLM）生成学生回复
#   阶段二（只需 CPU + 教师 vLLM API）：打标签、PPL计算、分组、教师续写、统计
#
# 用法：
#   bash scripts/experiments/run_exp2.sh
#
# 如果学生生成已完成，可以跳过阶段一，直接运行阶段二：
#   SKIP_STUDENT_GEN=1 bash scripts/experiments/run_exp2.sh

set -e

# ============================================================
# 环境配置
# ============================================================
export PATH="./conda/envs/verl_opd_topk/bin:$PATH"
export PYTHONPATH="./verl-distillation-ori:$PYTHONPATH"

PROJECT_PATH=./verl-distillation-ori
cd $PROJECT_PATH

# ============================================================
# 参数配置
# ============================================================

# 数据
DATA_PATH="$PROJECT_PATH/data/deepmath_new/deepmath_new_train.parquet"
SAMPLE_SIZE=2000
OUTPUT_DIR="$PROJECT_PATH/exp2_results"

# 学生模型（阶段一：main_generation.py 使用）
STUDENT_MODEL_PATH="./Models/DeepSeek-R1-Distill-Qwen-1.5B"
GPUS_PER_NODE=1

# 教师模型（阶段二：vLLM API 使用）
TEACHER_IP_POOL="xx.xxx.xx.xxx"
TEACHER_MODEL_NAME="Skywork-OR1-7B"

# 分桶参数
N_BINS=4
SPLIT_METHOD="quartile"

# 续写参数
N_CONTINUATIONS=4
TEACHER_TEMPERATURE=0.6
TRUNCATE_RATIO=0.6
MAX_TOKENS=32768

# 跳过标志（已完成的步骤可设为 1 跳过）
SKIP_STUDENT_GEN=${SKIP_STUDENT_GEN:-0}   # 1=跳过学生生成（已有 student_gen.parquet）
SKIP_LABEL=${SKIP_LABEL:-0}               # 1=跳过打标签（已有 student_gen_labeled.parquet）
SKIP_PPL=${SKIP_PPL:-0}                   # 1=跳过PPL计算（已有 wrong_samples_with_ppl.parquet）

SAMPLE_PATH="$OUTPUT_DIR/train_sample.parquet"
STUDENT_GEN_PATH="$OUTPUT_DIR/student_gen.parquet"

echo "======================================================"
echo "实验二：证明盲目蒸馏会传递教师幻觉"
echo "======================================================"
echo "数据: $DATA_PATH (采样 $SAMPLE_SIZE 条)"
echo "学生模型: $STUDENT_MODEL_PATH"
echo "教师模型: $TEACHER_MODEL_NAME"
echo "分桶: ${N_BINS}组(${SPLIT_METHOD}), 续写${N_CONTINUATIONS}次"
echo "输出目录: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# ============================================================
# 阶段一 Step 0：采样训练集子集
# ============================================================
if [ "$SKIP_STUDENT_GEN" != "1" ]; then
    echo "------------------------------------------------------"
    echo "阶段一 Step 0: 采样训练集子集（${SAMPLE_SIZE} 条）"
    echo "------------------------------------------------------"
    python3 scripts/experiments/exp2_teacher_hallucination.py \
        --step prepare_data \
        --data_path "$DATA_PATH" \
        --sample_size $SAMPLE_SIZE \
        --output_dir "$OUTPUT_DIR"
    echo ""

    # ============================================================
    # 阶段一 Step 1：main_generation.py 生成学生回复（需要 GPU）
    # ============================================================
    echo "------------------------------------------------------"
    echo "阶段一 Step 1: 学生模型生成回复（Ray + FSDP + vLLM）"
    echo "------------------------------------------------------"
    python3 -m verl.trainer.main_generation \
        --config-path "$PROJECT_PATH/scripts/experiments" \
        --config-name exp2_generation \
        data.path="$SAMPLE_PATH" \
        data.output_path="$STUDENT_GEN_PATH" \
        model.path="$STUDENT_MODEL_PATH" \
        trainer.n_gpus_per_node=$GPUS_PER_NODE
    echo "学生生成完成: $STUDENT_GEN_PATH"
    echo ""
else
    echo "[跳过阶段一] 使用已有学生生成结果: $STUDENT_GEN_PATH"
    echo ""
fi

# ============================================================
# 阶段二：教师分析（不需要 GPU）
# ============================================================
echo "------------------------------------------------------"
echo "阶段二: 教师分析（打标签 + PPL + 分组 + 续写 + 统计）"
echo "------------------------------------------------------"

# 构造跳过参数
SKIP_ARGS=""
if [ "$SKIP_LABEL" == "1" ]; then
    SKIP_ARGS="$SKIP_ARGS --skip_label"
fi
if [ "$SKIP_PPL" == "1" ]; then
    SKIP_ARGS="$SKIP_ARGS --skip_ppl"
fi

python3 scripts/experiments/exp2_teacher_hallucination.py \
    --step analyze \
    --student_gen_path "$STUDENT_GEN_PATH" \
    --student_model_path "$STUDENT_MODEL_PATH" \
    --teacher_ip_pool "$TEACHER_IP_POOL" \
    --teacher_model_name "$TEACHER_MODEL_NAME" \
    --n_bins $N_BINS \
    --split_method "$SPLIT_METHOD" \
    --truncate_ratio $TRUNCATE_RATIO \
    --n_continuations $N_CONTINUATIONS \
    --teacher_temperature $TEACHER_TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --output_dir "$OUTPUT_DIR" \
    $SKIP_ARGS

echo ""
echo "======================================================"
echo "实验完成！结果保存在 $OUTPUT_DIR"
echo "  train_sample.parquet                 采样的训练子集"
echo "  student_gen.parquet                  学生生成结果"
echo "  student_gen_labeled.parquet          打标签后的结果"
echo "  wrong_samples_with_ppl.parquet       错误样本 + 教师PPL + 截断前缀"
echo "  teacher_continuation_results.parquet 教师续写结果 + 恢复率"
echo "  summary_stats.json                   最终统计（四组恢复率对比）"
echo "======================================================"
