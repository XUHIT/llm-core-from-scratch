# 001 Linear Layer

## 目标

手写线性层的前向传播和反向传播，并用 PyTorch autograd 验证结果。

这一题不使用随机输入。所有数字固定，方便手算：

```text
x:      [B, in_features]  = [2, 3]
W:      [in_features, out_features] = [3, 2]
b:      [out_features] = [2]
y:      [B, out_features] = [2, 2]
dY:     [B, out_features] = [2, 2]
```

## 前向传播

$$y = xW + b$$

其中：

- 输入：$x \in \mathbb{R}^{B \times d_{in}}$
- 权重：$W \in \mathbb{R}^{d_{in} \times d_{out}}$
- 偏置：$b \in \mathbb{R}^{d_{out}}$
- 输出：$y \in \mathbb{R}^{B \times d_{out}}$

NumPy 实现采用数学中更直观的权重形状：

```text
W.shape = [in_features, out_features]
```

PyTorch 的 `nn.Linear` 内部存储方式不同：

```text
weight.shape = [out_features, in_features]
```

所以 PyTorch 里等价公式是：

$$y = xW^T + b$$

## 反向传播

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} W^T, \quad \frac{\partial L}{\partial W} = x^T \frac{\partial L}{\partial y}, \quad \frac{\partial L}{\partial b} = \sum_{batch} \frac{\partial L}{\partial y}$$

也就是：

```text
dX = dY @ W.T
dW = X.T @ dY
db = dY.sum(axis=0)
```

## 为什么这个例子适合练习

- 数字小，可以手算。
- shape 明确，能看清楚矩阵乘法方向。
- NumPy 手写 backward。
- PyTorch 用 autograd 验证 `dX / dW / db`。
- 不依赖真实数据集，不涉及训练效果。

## 文件

```text
README.md       # 公式、shape、推导说明
numpy_impl.py   # NumPy forward/backward 手写实现
torch_impl.py   # PyTorch autograd 对照实现
```

## 运行

```bash
python algorithms/p0_fundamentals/001_linear_layer/numpy_impl.py
python algorithms/p0_fundamentals/001_linear_layer/torch_impl.py
```

两个脚本应得到相同的 forward 输出和梯度。

## 常见坑

- 把 NumPy 的 `W` 和 PyTorch 的 `weight` 形状混淆。
- 忘记 bias 在 batch 维度上广播。
- 计算 `dW` 时写成 `dY.T @ X`，导致形状变成 `[out, in]`。
- 对三维输入 `[B, T, D]` 计算 backward 时，忘记把前面的维度展平再求 `dW/db`。
