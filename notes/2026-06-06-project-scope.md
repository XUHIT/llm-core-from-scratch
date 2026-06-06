# 2026-06-06 项目范围

## 项目名

`llm-core-from-scratch`

## 一句话定位

NumPy 与 PyTorch 双实现的大语言模型核心模块。

## 实现顺序

1. 张量基础、线性层、Embedding。
2. LayerNorm / RMSNorm。
3. Softmax、交叉熵、采样。
4. Self-Attention、Causal Mask、KV Cache。
5. RoPE 位置编码。
6. MLP / SwiGLU。
7. Transformer Block。
8. Mini GPT 推理与最小训练循环。

## 约定

- NumPy 实现优先讲清楚公式和维度变化。
- PyTorch 实现优先能训练、能上 GPU、能和 NumPy 对齐。
- 每个模块尽量配一组固定输入的数值对齐测试。
