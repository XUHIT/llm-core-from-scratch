# llm-core-from-scratch

面向面试手撕和底层机制理解的大语言模型核心模块实现。

目标：逐模块手写 LLM 关键组件，每个模块配知识点讲解、shape 变化、代码解析和数值验证。

## 实现原则

这个项目按题库层级选择实现方式，不是一律 PyTorch 先行。

### P0：NumPy first

基础手撕题优先用 NumPy 实现。

适用范围：

- Linear regression
- K-means
- Linear layer
- Embedding
- Softmax / Cross Entropy / KL / MSE
- BatchNorm / Dropout

原因：P0 的目标是看清楚每一步真实计算、shape 变化、broadcasting、forward/backward 推导。PyTorch 在这一层主要作为 reference checker，用来做 `allclose`、autograd 或 gradcheck 对照。

### P1：NumPy reference + PyTorch module

LLM 核心层采用 NumPy reference 和 PyTorch module 并重。

适用范围：

- LayerNorm / RMSNorm
- Causal mask
- Scaled dot-product attention
- MHA / GQA
- RoPE / ALiBi
- FFN / SwiGLU
- LM Head / Weight Tying

原因：NumPy 负责解释公式、维度和中间张量；PyTorch 负责模块化、梯度验证和后续组合成模型。

### P2：PyTorch first

推理、训练目标和最小模型闭环优先用 PyTorch 实现。

适用范围：

- KV Cache
- Decoder Block
- Tiny GPT forward
- Greedy / Top-k / Top-p / Beam Search
- AdamW / Gradient Clipping / Gradient Accumulation
- SFT / DPO / PPO / GRPO 等 loss

原因：P2 更接近模块组合和训练/推理流程，PyTorch 更适合验证工程行为。必要时再补 NumPy reference，帮助解释关键子步骤。

## 验证原则

第一阶段不使用真实数据集，不追求训练效果。

所有正确性验证优先使用：

- 小数字可读样例
- synthetic tensors
- 固定 random seed
- shape assertions
- NumPy / PyTorch 数值对齐

真实数据训练放到后续实验阶段。

## 本地草稿区

需要自己手敲代码和快速验证时，使用 `scratchpad/`。

`scratchpad/` 是本地练习区，已经被 `.gitignore` 忽略，不会提交到公开仓库。公开仓库只保留 `templates/scratchpad/` 作为模板；真正练习时把模板复制到 `scratchpad/`，然后在 `current.py` 里手写实现，在 `reference.py` 里放对照版本，用 `check.py` 快速运行。

详细说明见 `docs/local_scratchpad.md`。

## 目录

```text
algorithms/
├── README.md
├── p0_fundamentals/            # 001-013 基础组件
│   ├── 001_tensor_shape/
│   ├── 002_matrix_multiplication/
│   ├── 003_linear_regression/
│   ├── 004_k_means/
│   ├── 005_linear_layer/
│   ├── 006_embedding/
│   ├── 007_activations/
│   ├── 008_softmax/
│   ├── 009_cross_entropy/
│   ├── 010_kl_divergence/
│   ├── 011_mse_loss/
│   ├── 012_batchnorm/
│   └── 013_dropout/
├── p1_llm_core/                # 014-024 LLM 核心层
│   ├── 014_layernorm/
│   ├── 015_rmsnorm/
│   ├── 016_causal_mask/
│   ├── 017_scaled_dot_product_attention/
│   ├── 018_multi_head_attention/
│   ├── 019_grouped_query_attention/
│   ├── 020_rope/
│   ├── 021_ffn_mlp/
│   ├── 022_swiglu_ffn/
│   ├── 023_lm_head_weight_tying/
│   └── 024_alibi/
└── p2_inference_training/      # 025-035 推理、训练目标与模型闭环
    ├── 025_kv_cache/
    ├── 026_decoder_block/
    ├── 027_tiny_gpt_forward/
    ├── 028_greedy_temperature_sampling/
    ├── 029_top_k_sampling/
    ├── 030_top_p_sampling/
    ├── 031_adamw/
    ├── 032_gradient_clipping/
    ├── 033_sft_loss/
    ├── 034_beam_search/
    └── 035_gradient_accumulation/

每道题文件夹默认包含：
README.md      # 公式、shape、手推、常见坑
numpy_impl.py  # NumPy 手写实现
torch_impl.py  # PyTorch 对照实现

docs/
└── interview_scratch_foundations.md   # 完整题库与路线图

experiments/    # 实验代码和配置
notes/          # 实验笔记
results/        # 图表、指标、运行结果摘要
tests/          # 测试（形状 + 数值 + 双实现对齐，后补）
```

## 安装

```bash
pip install -e ".[torch,dev]"
```
