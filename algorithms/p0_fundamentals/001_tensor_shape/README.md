# 001 Tensor Shape

## 目标

这一题练的是 LLM 代码里最常见的张量维度变化，不涉及模型参数和训练。

固定使用一个小张量：

```text
B = 2  batch size
T = 3  sequence length
D = 4  hidden size
H = 2  attention heads
Dh = D / H = 2  head dimension
```

核心输入：

```text
x.shape = [B, T, D] = [2, 3, 4]
```

## 必须掌握的 shape

### 1. 展平 batch 和 time

很多线性层、loss 或统计操作会把 `[B, T, D]` 看成 `[B*T, D]`：

```text
[B, T, D] -> [B*T, D]
[2, 3, 4] -> [6, 4]
```

只要元素总数不变，就可以 reshape：

```text
2 * 3 * 4 = 6 * 4 = 24
```

### 2. 拆分 attention heads

多头注意力常把 hidden size 拆成 `H` 个 head：

```text
[B, T, D] -> [B, T, H, Dh] -> [B, H, T, Dh]
[2, 3, 4] -> [2, 3, 2, 2] -> [2, 2, 3, 2]
```

第二步 transpose 的目的，是让 attention 分数方便写成：

```text
scores.shape = [B, H, T, T]
```

### 3. 合并 attention heads

attention 输出通常要从 `[B, H, T, Dh]` 合回 `[B, T, D]`：

```text
[B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
[2, 2, 3, 2] -> [2, 3, 2, 2] -> [2, 3, 4]
```

### 4. Broadcasting

LLM 中很常见的广播：

```text
x:        [B, T, D]
bias:     [D]
x + bias: [B, T, D]
```

```text
x:        [B, T, D]
pos_emb:  [T, D]
x + pos:  [B, T, D]
```

```text
scores:   [B, H, T, T]
mask:     [T, T]
mask 后:  [B, H, T, T]
```

## NumPy 和 PyTorch 的差异

NumPy 和 PyTorch 的 shape 逻辑基本一致。

但 PyTorch 里 `transpose` / `permute` 后，张量经常变成 non-contiguous。此时：

```python
x.view(...)
```

可能报错。更稳妥的写法是：

```python
x.contiguous().view(...)
```

或者：

```python
x.reshape(...)
```

本题 PyTorch 版本会显式使用 `contiguous().view(...)`，方便理解底层内存布局问题。

## 文件

```text
README.md       # shape 说明
numpy_impl.py   # NumPy 维度变化与广播
torch_impl.py   # PyTorch 维度变化与广播
```

## 运行

```bash
python3 algorithms/p0_fundamentals/001_tensor_shape/numpy_impl.py
python3 algorithms/p0_fundamentals/001_tensor_shape/torch_impl.py
```

两个脚本都应该打印关键张量的 shape，并通过所有 assert。

## 常见坑

- 把 `[B, T, D]` 和 `[T, B, D]` 混用。
- 拆 head 后忘记 transpose，导致 attention 的矩阵乘法维度不对。
- 合并 head 时忘记先把 `[B, H, T, Dh]` 转回 `[B, T, H, Dh]`。
- 只看 reshape 后 shape 对了，但元素顺序已经错了。
- PyTorch 里 transpose 后直接 `view`，遇到 non-contiguous 报错。
- broadcasting 时没有确认从右往左对齐维度。
