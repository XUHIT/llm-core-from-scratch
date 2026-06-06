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

## 目录

```
algorithms/
├── p0_fundamentals/            # 基础组件
│   ├── 001_linear_layer/
│   │   ├── README.md           # 公式、shape、手推、常见坑
│   │   ├── numpy_impl.py       # NumPy 手写主实现
│   │   └── torch_impl.py       # PyTorch 对照实现
│   └── ...
├── p1_llm_core/                # LLM 核心层
│   └── ...
└── p2_inference_training/      # 推理、训练目标与模型闭环
    └── ...

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
