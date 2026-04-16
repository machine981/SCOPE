# SCOPE

SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting

📑 [论文](https://arxiv.org/pdf/2604.10688)

## 项目概述

SCOPE 是一个双路径自适应训练框架，将在线蒸馏中的轨迹按正确性分为两条互补的监督路径：

- **教师困惑度加权的 Reverse-KL 蒸馏**：用于错误轨迹，优先选择可靠的正向指导
- **学生困惑度加权的 MLE**：用于正确轨迹，强化能力边界上未被充分探索的推理路径

## 核心结果

SCOPE 在 6 个数学推理基准测试中相比基线方法取得 **11.42%** 的 Avg@32 相对提升和 **7.30%** 的 Pass@32 相对提升。

### 主要结果 (Teacher: Skywork-OR1-Math-7B → Student: DeepSeek-R1-Distill-Qwen-1.5B)

| Benchmark | Avg@32 | Pass@32 | vs OPD  |
| --------- | ------ | ------- | ------- |
| AIME24    | 42.7   | 77.9    | +6.22%  |
| AIME25    | 30.4   | 50.9    | +5.19%  |
| AMC23     | 80.9   | 97.2    | +6.59%  |
| MATH500   | 89.8   | 97.9    | +0.90%  |
| Minerva   | 37.8   | 55.1    | +8.31%  |
| Olympiad  | 49.7   | 70.9    | +10.69% |

## 快速开始

### 1. 启动 VLLM 服务

```bash
bash deploy_vllm.sh
```

**`deploy_vllm.sh` 关键配置**：

| 参数                   | 说明                 | 默认值                                          |
| ---------------------- | -------------------- | ----------------------------------------------- |
| `model_name_or_path` | 模型路径             | `./Models/Skywork-OR1-7B`                     |
| `served_model_name`  | API 服务中的模型名称 | `Skywork-OR1-7B`                              |
| `--api-key`          | API 认证密钥         | `xxx`（需与 api_interface 中的 key 保持一致） |

### 2. 配置实验脚本

在 `run_experiment_distill_1_5b.sh` 中设置：

```bash
TEACHER_MODEL_NAME=Skywork-OR1-7B  # 必须与 deploy_vllm.sh 中的 served_model_name 一致
IP_POOL="['xx.xxx.x.xx','...']"      # VLLM 服务节点 IP 列表
```

**API Key 一致性**：VLLM 服务的 `--api-key` 必须与以下文件中的 `api_key` 保持一致：

- `scripts/experiments/exp2_teacher_hallucination.py`
- `scripts/experiments/exp1_teacher_avg4.py`

### 3. 运行训练

```bash
bash run_experiment_distill_1_5b.sh
```

---

## 训练脚本参数说明

### 模型配置

| 参数                   | 说明                                  | 默认值                                     |
| ---------------------- | ------------------------------------- | ------------------------------------------ |
| `POLICY_MODEL_PATH`  | 学生模型路径                          | `./Models/DeepSeek-R1-Distill-Qwen-1.5B` |
| `TEACHER_MODEL_NAME` | 教师模型名称（VLLM 服务中注册的名字） | `Skywork-OR1-7B`                         |
| `IP_POOL`            | VLLM 服务节点 IP 列表                 | `['xx.xxx.x.xx','...']`                  |

### 数据配置

| 参数                    | 说明         | 默认值                                                                   |
| ----------------------- | ------------ | ------------------------------------------------------------------------ |
| `TRAIN_DATA`          | 训练数据路径 | `./verl-distillation-ori/data/deepmath_new/deepmath_new_train.parquet` |
| `VAL_DATA`            | 验证数据路径 | `./verl-distillation-ori/data/aime/test.parquet`                       |
| `MAX_PROMPT_LENGTH`   | 最大提示长度 | `2048`                                                                 |
| `MAX_RESPONSE_LENGTH` | 最大回复长度 | `12288`                                                                |

### 蒸馏配置

| 参数                       | 说明                  | 默认值                        |
| -------------------------- | --------------------- | ----------------------------- |
| `DISTILL_ENABLE`         | 是否启用蒸馏          | `True`                      |
| `DISTILL_SIGNAL_IN_LOSS` | 蒸馏信号是否进入 loss | `False`                     |
| `DISTILL_LOSS_TYPE`      | 蒸馏损失类型          | `low_var_kl`（可选 `kl`） |

### SCOPE 双路径加权配置

| 参数                              | 说明                       | 默认值    |
| --------------------------------- | -------------------------- | --------- |
| `USE_SCOPE_DUAL_PATH_WEIGHTING` | 是否启用 SCOPE 双路径加权  | `True`  |
| `SCOPE_TAU`                     | 加权温度参数               | `1`     |
| `SCOPE_USE_SEQ_WEIGHTS`         | 是否使用序列级权重         | `True`  |
| `USE_STUDENT_PATH_WEIGHTS`      | 是否使用学生路径权重       | `True`  |
| `USE_TEACHER_PATH_WEIGHTS`      | 是否使用教师路径权重       | `True`  |
| `STUDENT_PATH_PPL_POSITIVE`     | 学生路径：PPL 越高权重越高 | `True`  |
| `TEACHER_PATH_PPL_POSITIVE`     | 教师路径：PPL 越高权重越低 | `False` |
