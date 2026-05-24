# SCOPE

## Quick Start

### 1. Deploy VLLM Service

```bash
bash deploy_vllm.sh
```

**Key configurations in `deploy_vllm.sh`**:

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `model_name_or_path` | Model path | `./Models/Skywork-OR1-7B` |
| `served_model_name` | Model name in API | `Skywork-OR1-7B` |
| `--api-key` | API authentication key | `xxx` (must match `verl/utils/api_interface.py`) |

### 2. Configure Experiment Scripts

Set the following in `run_experiment_distill_1_5b.sh`:

```bash
TEACHER_MODEL_NAME=Skywork-OR1-7B  # Must match served_model_name in deploy_vllm.sh
IP_POOL="['xx.xxx.x.xx','...']"    # VLLM service node IP list
```

**API Key Consistency**: The `--api-key` in `deploy_vllm.sh` must match the `api_key` in `verl/utils/api_interface.py`.

### 3. Run Training

```bash
bash run_experiment_distill_1_5b.sh
```

---

## Training Parameters

### Model Configuration

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `POLICY_MODEL_PATH` | Student model path | `DeepSeek-R1-Distill-Qwen-1.5B` |
| `TEACHER_MODEL_NAME` | Teacher model name (as registered in VLLM) | `Skywork-OR1-7B` |
| `IP_POOL` | VLLM service node IP list | `['xx.xxx.x.xx','...']` |

### Data Configuration

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `TRAIN_DATA` | Training data path | `./verl-distillation-ori/data/deepmath_new/deepmath_new_train.parquet` |
| `VAL_DATA` | Validation data path | `./verl-distillation-ori/data/aime/test.parquet` |
| `MAX_PROMPT_LENGTH` | Max prompt length | `2048` |
| `MAX_RESPONSE_LENGTH` | Max response length | `12288` |

### SCOPE Dual-Path Configuration

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `USE_SCOPE_DUAL_PATH_WEIGHTING` | Enable SCOPE dual-path weighting | `True` |
| `SCOPE_TAU` | Weight temperature parameter | `1` |
| `SCOPE_USE_SEQ_WEIGHTS` | Use sequence-level weights | `True` |
| `USE_STUDENT_PATH_WEIGHTS` | Use student path weights | `True` |
| `USE_TEACHER_PATH_WEIGHTS` | Use teacher path weights | `True` |
| `STUDENT_PATH_PPL_POSITIVE` | Student path: higher PPL → higher weight | `True` |
| `TEACHER_PATH_PPL_POSITIVE` | Teacher path: higher PPL → lower weight | `False` |
