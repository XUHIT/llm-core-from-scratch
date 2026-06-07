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

## 函数用法速记

### `torch.arange`

生成连续数字张量，像 Python 的 `range`，但返回 Tensor。

```python
x = torch.arange(24, dtype=torch.float64)
```

```text
x = [0, 1, 2, ..., 23]
x.shape = [24]
```

```python
x = torch.arange(B * T * D, dtype=torch.float64)
x = x.reshape(B, T, D)
```

本题用它造小数字，方便检查 reshape / transpose 后元素顺序。

### `reshape`

改变张量形状，但元素总数必须不变。

```python
x = x.reshape(2, 3, 4)
```

```text
[24] -> [2, 3, 4]
```

在本题里最常见的是：

```python
x_2d = x.reshape(B * T, D)
```

```text
[B, T, D] -> [B*T, D]
```

### `transpose`

交换两个维度的位置。

```python
x = x.transpose(1, 2)
```

```text
[B, T, H, Dh] -> [B, H, T, Dh]
```

### `contiguous`

`transpose` 后张量内存通常不连续，接 `view` 前先用它。

```python
x = x.transpose(1, 2).contiguous()
x = x.view(B, T, H * Dh)
```

```text
[B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
```

### Broadcasting

PyTorch 从右往左对齐维度，自动扩展缺失维度。

```python
y = x + bias
```

```text
x.shape    = [B, T, D]
bias.shape = [D]
y.shape    = [B, T, D]
```

```python
y = x + pos_emb
```

```text
x.shape       = [B, T, D]
pos_emb.shape = [T, D]
y.shape       = [B, T, D]
```

### `torch.triu`

取上三角，常用来构造 causal mask。

```python
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
```

```text
mask.shape = [T, T]
```

`diagonal=1`：主对角线保留，只 mask 未来位置。

### `masked_fill`

把 mask 为 `True` 的位置替换成指定值。

```python
masked_scores = scores.masked_fill(mask[None, None, :, :], -1e9)
```

```text
scores.shape = [B, H, T, T]
mask.shape   = [T, T]
输出 shape   = [B, H, T, T]
```

```text
[T, T] -> [1, 1, T, T]
```

## 文件

```text
README.md       # shape 说明
torch_impl.py   # PyTorch 维度变化与广播
numpy_impl.py   # NumPy 辅助对照
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
