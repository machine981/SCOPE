<h1 align="center">
SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting
</h1>

<div align="center">
  <a href='https://arxiv.org/pdf/2604.10688'><img src='https://img.shields.io/badge/arXiv-2604.10688-red?logo=arXiv'></a>  &nbsp;
  <a href="https://huggingface.co/papers/2604.10688"><img src='https://img.shields.io/badge/HuggingFace-SCOPE-FF9B9E?logo=huggingface'></img></a>  &nbsp;
  <a href="https://github.com/machine981/SCOPE"><img src="https://img.shields.io/badge/GitHub-SCOPE-94c320?logo=github"></a> &nbsp;
  <br>
</div>

![SCOPE Overview](image/SCOPE_Overview.png)

---

## 🔥 新闻

- **[2026.04]** SCOPE 模型权重已发布！
  - <a href="https://huggingface.co/Machine981/SCOPE-Qwen3-1.7B"><img src="https://img.shields.io/badge/HuggingFace-SC--OPE--Qwen3--1.7B-FF9B9E?logo=huggingface"></a>
  - <a href="https://huggingface.co/Machine981/SCOPE-Deepseek-R1-Distill-Qwen-1.5B"><img src="https://img.shields.io/badge/HuggingFace-SC--OPE--Deepseek--R1--Distill--Qwen--1.5B-FF9B9E?logo=huggingface"></a>
- **[2026.04]** 论文已在 arXiv 发布！

---

## 📖 概述

在线蒸馏（On-Policy Distillation, OPD）通过引入教师模型提供的密集 token 级 KL 监督来缓解对齐差距，但通常对所有在线 rollout 均匀应用这种监督，忽视了信号质量的根本差异。

**现有 OPD 的局限性：**
- **多样性退化**：正确路径被同等强化，减少了能力边界的探索
- **纠正效率低下**：嘈杂的教师信号误导错误轨迹

**SCOPE 解决方案：**
我们提出了信号校准在线蒸馏增强（SCOPE），一种双路径自适应训练框架，将在线 rollout 按正确性路由到两条互补的监督路径。

---

## 📝 摘要

在线蒸馏（OPD）通过引入教师模型提供的密集 token 级 KL 监督来缓解对齐差距，但通常对所有在线 rollout 均匀应用这种监督，忽视了信号质量的根本差异。我们提出了信号校准在线蒸馏增强（SCOPE），一种双路径自适应训练框架，将在线 rollout 按正确性路由到两条互补的监督路径。对于错误轨迹，SCOPE 执行教师困惑度加权的 KL 蒸馏，优先选择教师表现出真正纠正能力的样本，同时降低不可靠指导的权重。对于正确轨迹，它应用学生困惑度加权的 MLE，将强化集中在能力边界上的低置信度样本，而不是过度强化已掌握的样本。两条路径都采用组级归一化来自适应校准权重分布，考虑了跨提示的固有难度差异。在六个推理基准上的广泛实验表明，SCOPE 在 Avg@32 上实现了 11.42% 的平均相对提升，在 Pass@32 上实现了 7.30% 的平均相对提升，展示了其持续的有效性。

---

## 🏆 主要贡献

- **对 OPD 信号质量异质性的实证分析：** 揭示了教师和学生困惑度可可靠地预测错误轨迹上的纠正能力和正确轨迹上能力边界的样本。

- **SCOPE 双路径自适应框架：** 按正确性路由 rollout，将错误轨迹导向教师困惑度加权的 OPD，将正确轨迹导向学生困惑度加权的 MLE。

- **广泛的实验验证：** 在六个推理基准上相比基线在 Avg@32 上实现 11.42% 的相对提升，在 Pass@32 上实现 7.30% 的相对提升。

---

## 📖 方法

### SCOPE 框架

SCOPE 是一个双路径自适应训练框架，将在线 rollout 按正确性路由到两条互补的监督路径：

| 路径 | 轨迹 | 方法 | 目标 |
|------|------|------|------|
| **学生路径** | 正确 (Ω_c) | 困惑度加权的 MLE | 强化能力边界上非常规有效路径 |
| **教师路径** | 错误 (Ω_w) | 困惑度加权的 KL 蒸馏 | 过滤上下文诱导的噪声，优先选择可靠指导 |

### 权重公式

**学生引导权重（用于正确轨迹 Ω_c）：**

$$w_i^{stu} = \frac{\text{PPL}_S(y_i|x)^{1/\tau}}{\sum_{j \in \Omega_c} \text{PPL}_S(y_j|x)^{1/\tau}}$$

使用基于困惑度的加权放大"能力边界的非常规有效路径"。

**教师引导权重（用于错误轨迹 Ω_w）：**

$$w_i^{tea} = \frac{\text{PPL}_T(y_i|x)^{-1/\tau}}{\sum_{j \in \Omega_w} \text{PPL}_T(y_j|x)^{-1/\tau}}$$

通过降低高教师困惑度实例的权重来过滤"上下文诱导的噪声"。

### 关键洞察

在每个提示的轨迹组内，SCOPE 应用**组级基于困惑度的归一化**来自适应校准权重分布，考虑了跨提示的固有难度差异。

### 总体目标

联合优化组合的 SCOPE 目标：

$$\mathcal{L}_{SCOPE} = \sum_{i \in \Omega_c} w_i^{stu} \cdot \mathcal{L}_{MLE} + \sum_{i \in \Omega_w} w_i^{tea} \cdot \mathcal{L}_{OPD}$$

---

## 📊 主要结果

### 数学推理 (Teacher: Skywork-OR1-Math-7B → Student: DeepSeek-R1-Distill-Qwen-1.5B)

| Benchmark | Avg@32 | Pass@32 | vs OPD |
| --------- | ------ | ------- | ------ |
| AIME24 | 42.7 | 77.9 | +6.22% |
| AIME25 | 30.4 | 50.9 | +5.19% |
| AMC23 | 80.9 | 97.2 | +6.59% |
| MATH500 | 89.8 | 97.9 | +0.90% |
| Minerva | 37.8 | 55.1 | +8.31% |
| Olympiad | 49.7 | 70.9 | +10.69% |

**关键发现：**
- **11.42%** Avg@32 相对提升
- **7.30%** Pass@32 相对提升
- 相比标准 OPD 在各基准上平均 **+5.54%** 提升

---

## ⚡ 快速开始

### 1. 部署 VLLM 服务

```bash
bash deploy_vllm.sh
```

**`deploy_vllm.sh` 关键配置**：

| 参数 | 说明 | 默认值 |
| --------- | ----------- | ------- |
| `model_name_or_path` | 模型路径 | `./Models/Skywork-OR1-7B` |
| `served_model_name` | API 服务中的模型名称 | `Skywork-OR1-7B` |
| `--api-key` | API 认证密钥 | `xxx`（需与 `verl/utils/api_interface.py` 中的 key 保持一致） |

### 2. 配置实验脚本

在 `run_experiment_distill_1_5b.sh` 中设置：

```bash
TEACHER_MODEL_NAME=Skywork-OR1-7B  # 必须与 deploy_vllm.sh 中的 served_model_name 一致
IP_POOL="['xx.xxx.x.xx','...']"      # VLLM 服务节点 IP 列表
```

**API Key 一致性**：`deploy_vllm.sh` 中的 `--api-key` 必须与 `verl/utils/api_interface.py` 中的 `api_key` 保持一致。

### 3. 运行训练

```bash
bash run_experiment_distill_1_5b.sh
```

---

## 🔧 训练参数

### 模型配置

| 参数 | 说明 | 默认值 |
| --------- | ----------- | ------- |
| `POLICY_MODEL_PATH` | 学生模型路径 | `DeepSeek-R1-Distill-Qwen-1.5B` |
| `TEACHER_MODEL_NAME` | 教师模型名称（VLLM 服务中注册的名字） | `Skywork-OR1-7B` |
| `IP_POOL` | VLLM 服务节点 IP 列表 | `['xx.xxx.x.xx','...']` |

### 数据配置

| 参数 | 说明 | 默认值 |
| --------- | ----------- | ------- |
| `TRAIN_DATA` | 训练数据路径 | `./verl-distillation-ori/data/deepmath_new/deepmath_new_train.parquet` |
| `VAL_DATA` | 验证数据路径 | `./verl-distillation-ori/data/aime/test.parquet` |
| `MAX_PROMPT_LENGTH` | 最大提示长度 | `2048` |
| `MAX_RESPONSE_LENGTH` | 最大回复长度 | `12288` |

### SCOPE 双路径配置

| 参数 | 说明 | 默认值 |
| --------- | ----------- | ------- |
| `USE_SCOPE_DUAL_PATH_WEIGHTING` | 启用 SCOPE 双路径加权 | `True` |
| `SCOPE_TAU` | 加权温度参数 | `1` |
| `SCOPE_USE_SEQ_WEIGHTS` | 使用序列级权重 | `True` |
| `USE_STUDENT_PATH_WEIGHTS` | 使用学生路径权重 | `True` |
| `USE_TEACHER_PATH_WEIGHTS` | 使用教师路径权重 | `True` |
| `STUDENT_PATH_PPL_POSITIVE` | 学生路径：PPL 越高权重越高 | `True` |
| `TEACHER_PATH_PPL_POSITIVE` | 教师路径：PPL 越高权重越低 | `False` |

---

## 🤝 致谢

本工作基于 [verl](https://github.com/volcengine/verl) 和在线蒸馏范式构建，感谢他们对研究社区的贡献。

## 🔗 引用

如果我们的工作对您有帮助，请考虑引用：

```bibtex
@article{scope2026,
  title={SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting},
  author={Zheng, Binbin and Ma, Xing and Liang, Yiheng and Ruan, Jingqing and Fu, Xiaoliang and Lin, Kepeng and Zhu, Benchang and Zeng, Ke and Cai, Xunliang},
  journal={arXiv preprint arXiv:2604.10688},
  year={2026}
}
```

## 📝 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
