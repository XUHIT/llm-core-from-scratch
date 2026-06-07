# Algorithms

这里是项目主线：每道面试手撕题一个独立文件夹。

默认结构：

```text
xxx_algorithm_name/
├── README.md
├── torch_impl.py
└── numpy_impl.py
```

- `README.md`: 公式、shape 变化、手推重点、常见坑。
- `torch_impl.py`: PyTorch 主实现。每题先保证它可运行、可读、可验证。
- `numpy_impl.py`: NumPy 辅助对照。用于解释公式、展开小数字计算或做双实现数值对齐。

第一阶段只使用小数字样例和 synthetic tensors，不使用真实数据集。所有题目默认 PyTorch first。

## 当前骨架

- `p0_fundamentals/001-013`: shape、矩阵乘法、线性回归、K-means、线性层、Embedding、激活函数、Softmax、CE、KL、MSE、BatchNorm、Dropout。
- `p1_llm_core/014-024`: LayerNorm、RMSNorm、causal mask、attention、MHA、GQA、RoPE、FFN、SwiGLU、LM Head、ALiBi。
- `p2_inference_training/025-035`: KV cache、decoder block、Tiny GPT、采样、AdamW、梯度裁剪、SFT loss、beam search、梯度累积。

LoRA、QLoRA、DPO、GRPO、PPO、MoE、MTP 等先保留在路线图中，后续进入第二阶段再建实现骨架。
