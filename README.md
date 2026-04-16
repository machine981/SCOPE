# SCOPE

SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting

📑 [Paper](https://arxiv.org/pdf/2604.10688)

## Overview

SCOPE is a dual-path adaptive training framework for on-policy distillation that routes rollouts by correctness into two complementary supervision paths:

- **Teacher perplexity-weighted reverse-KL distillation** for incorrect trajectories to prioritize reliable corrective guidance
- **Student perplexity-weighted MLE** for correct trajectories to reinforce under-explored reasoning paths at the capability boundary

## Key Results

SCOPE achieves **11.42%** relative improvement in Avg@32 and **7.30%** in Pass@32 over competitive baselines on 6 mathematical reasoning benchmarks.

### Main Results (Teacher: Skywork-OR1-Math-7B → Student: DeepSeek-R1-Distill-Qwen-1.5B)

| Benchmark | Avg@32 | Pass@32 | vs OPD  |
| --------- | ------ | ------- | ------- |
| AIME24    | 42.7   | 77.9    | +6.22%  |
| AIME25    | 30.4   | 50.9    | +5.19%  |
| AMC23     | 80.9   | 97.2    | +6.59%  |
| MATH500   | 89.8   | 97.9    | +0.90%  |
| Minerva   | 37.8   | 55.1    | +8.31%  |
| Olympiad  | 49.7   | 70.9    | +10.69% |

## Quick Start

### 1. Deploy VLLM Service

```bash
bash deploy_vllm.sh
```

**Key configurations in `deploy_vllm.sh`**:

| Parameter              | Description            | Default                            |
| ---------------------- | ---------------------- | ---------------------------------- |
| `model_name_or_path` | Model path             | `./Models/Skywork-OR1-7B`        |
| `served_model_name`  | Model name in API      | `Skywork-OR1-7B`                 |
| `--api-key`          | API authentication key | `xxx` (must match api_interface) |

### 2. Configure Experiment Scripts

Set the following in `run_experiment_distill_1_5b.sh`:

```bash
TEACHER_MODEL_NAME=Skywork-OR1-7B  # Must match served_model_name in deploy_vllm.sh
IP_POOL="['xx.xxx.x.xx','...']"    # VLLM service node IP list
```

**API Key Consistency**: The `--api-key` in `deploy_vllm.sh` must match the `api_key` in:

- `scripts/experiments/exp2_teacher_hallucination.py`
- `scripts/experiments/exp1_teacher_avg4.py`

### 3. Run Training

```bash
bash run_experiment_distill_1_5b.sh
```

---

## Training Script Parameters

### Model Configuration

| Parameter              | Description                                | Default                                    |
| ---------------------- | ------------------------------------------ | ------------------------------------------ |
| `POLICY_MODEL_PATH`  | Student model path                         | `./Models/DeepSeek-R1-Distill-Qwen-1.5B` |
| `TEACHER_MODEL_NAME` | Teacher model name (as registered in VLLM) | `Skywork-OR1-7B`                         |
| `IP_POOL`            | VLLM service node IP list                  | `['xx.xxx.x.xx','...']`                  |

### Data Configuration

| Parameter               | Description          | Default                                                                  |
| ----------------------- | -------------------- | ------------------------------------------------------------------------ |
| `TRAIN_DATA`          | Training data path   | `./verl-distillation-ori/data/deepmath_new/deepmath_new_train.parquet` |
| `VAL_DATA`            | Validation data path | `./verl-distillation-ori/data/aime/test.parquet`                       |
| `MAX_PROMPT_LENGTH`   | Max prompt length    | `2048`                                                                 |
| `MAX_RESPONSE_LENGTH` | Max response length  | `12288`                                                                |

### Distillation Configuration

| Parameter                  | Description                                     | Default                    |
| -------------------------- | ----------------------------------------------- | -------------------------- |
| `DISTILL_ENABLE`         | Enable distillation                             | `True`                   |
| `DISTILL_SIGNAL_IN_LOSS` | Whether distillation signal contributes to loss | `False`                  |
| `DISTILL_LOSS_TYPE`      | Distillation loss type                          | `low_var_kl` (or `kl`) |

### SCOPE Dual-Path Weighting Configuration

| Parameter                         | Description                               | Default   |
| --------------------------------- | ----------------------------------------- | --------- |
| `USE_SCOPE_DUAL_PATH_WEIGHTING` | Enable SCOPE dual-path weighting          | `True`  |
| `SCOPE_TAU`                     | Weight temperature parameter              | `1`     |
| `SCOPE_USE_SEQ_WEIGHTS`         | Use sequence-level weights                | `True`  |
| `USE_STUDENT_PATH_WEIGHTS`      | Use student path weights                  | `True`  |
| `USE_TEACHER_PATH_WEIGHTS`      | Use teacher path weights                  | `True`  |
| `STUDENT_PATH_PPL_POSITIVE`     | Student path: higher PPL → higher weight | `True`  |
| `TEACHER_PATH_PPL_POSITIVE`     | Teacher path: higher PPL → lower weight  | `False` |
