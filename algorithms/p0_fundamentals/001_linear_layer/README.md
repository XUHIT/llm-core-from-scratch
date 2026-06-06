# 001 Linear Layer

## 知识点

$$y = xW + b$$

- 输入 $x \in \mathbb{R}^{B \times d_{in}}$
- 权重 $W \in \mathbb{R}^{d_{in} \times d_{out}}$
- 偏置 $b \in \mathbb{R}^{1 \times d_{out}}$
- 输出 $y \in \mathbb{R}^{B \times d_{out}}$

### 反向传播

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} W^T, \quad \frac{\partial L}{\partial W} = x^T \frac{\partial L}{\partial y}, \quad \frac{\partial L}{\partial b} = \sum_{batch} \frac{\partial L}{\partial y}$$

### 权重初始化

Xavier uniform: $W \sim U\left(-\sqrt{1/d_{in}},\ \sqrt{1/d_{in}}\right)$

## 代码解析

- NumPy 主实现使用 $W \in \mathbb{R}^{d_{in} \times d_{out}}$，forward 做 `x @ W + b`
- `nn.Linear` 内部存储的是 $W^T$，PyTorch forward 做 `x @ weight.T + bias`
- 输入 `[B, T, D]` 时线性层自动作用在最后一维

## 文件

```text
README.md
numpy_impl.py
torch_impl.py
```

## 延伸

- Batch matmul 与线性层的等价关系
- LoRA 变体：冻结 $W$，训练低秩 $BA$ 作为增量
