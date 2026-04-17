<h1 align="center">
SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting
</h1>

<div align="center">
  <a href='https://arxiv.org/pdf/2604.10688'><img src='https://img.shields.io/badge/arXiv-2604.10688-red?logo=arXiv'></a>  &nbsp;
  <a href="https://github.com/machine981/SCOPE"><img src="https://img.shields.io/badge/GitHub-SCOPE-94c320?logo=github"></a> &nbsp;
  <br>
</div>

![SCOPE Overview](image/SCOPE_Overview.png)

---

## 🔥 News

- **[2026.04]** Paper released on arXiv!

## 📖 Method

### Motivation

On-policy distillation (OPD) suffers from **signal quality heterogeneity** — the same teacher signal has varying reliability across different trajectories. Standard OPD treats all trajectories uniformly, leading to:
- **Diversity degradation**: Correct paths reinforced equally, reducing exploration
- **Rectification inefficiency**: Noisy teacher signals misleading incorrect paths

### SCOPE Framework

SCOPE is a dual-path adaptive training framework that routes on-policy rollouts by correctness into two complementary supervision paths:

| Path | Trajectories | Method | Objective |
|------|-------------|--------|-----------|
| **Student Path** | Correct (Ω_c) | Perplexity-weighted MLE | Reinforce unconventional valid paths at capability boundary |
| **Teacher Path** | Incorrect (Ω_w) | Perplexity-weighted KL distillation | Filter out context-induced noise, prioritize reliable guidance |

### Key Insight

Within each prompt's trajectory group, SCOPE applies **group-level perplexity-based weighting** to amplify samples where the model is less confident (higher perplexity), addressing the diverse signal quality problem.

---

## 📊 Main Results

### Mathematical Reasoning (Teacher: Skywork-OR1-Math-7B → Student: DeepSeek-R1-Distill-Qwen-1.5B)

| Benchmark | Avg@32 | Pass@32 | vs OPD |
| --------- | ------ | ------- | ------ |
| AIME24 | 42.7 | 77.9 | +6.22% |
| AIME25 | 30.4 | 50.9 | +5.19% |
| AMC23 | 80.9 | 97.2 | +6.59% |
| MATH500 | 89.8 | 97.9 | +0.90% |
| Minerva | 37.8 | 55.1 | +8.31% |
| Olympiad | 49.7 | 70.9 | +10.69% |

**Key findings:**
- **11.42%** relative improvement in Avg@32
- **7.30%** relative improvement in Pass@32
- **+5.54%** average improvement over standard OPD across benchmarks

---

## ⚡ Quick Start

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

## 🔧 Training Parameters

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

---

## 🤝 Acknowledgements

This work builds upon [verl](https://github.com/volcengine/verl) and the on-policy distillation paradigm, with appreciation for their contributions to the research community.

## 🔗 Citation

If you find our work useful, please consider citing:

```bibtex
@article{scope2026,
  title={SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting},
  author={Zheng, Binbin and Ma, Xing and Liang, Yiheng and Ruan, Jingqing and Fu, Xiaoliang and Lin, Kepeng and Zhu, Benchang and Zeng, Ke and Cai, Xunliang},
  journal={arXiv preprint arXiv:2604.10688},
  year={2026}
}
```

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
