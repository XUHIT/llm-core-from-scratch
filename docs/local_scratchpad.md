# 本地草稿区

这个项目会把正式答案放在 `algorithms/` 下，每道题都有 `README.md`、`torch_impl.py` 和 `numpy_impl.py`。

但练习时需要一个更随意的地方：先自己手敲 PyTorch 版本，马上运行，看 shape 和数值是否正确。这个地方就是：

```text
scratchpad/
```

## 为什么不提交 scratchpad

`scratchpad/` 默认被 `.gitignore` 忽略。

原因很简单：

- 这里放的是临时练习代码。
- 可以随便改、随便删、随便写错。
- 不会污染公开 GitHub 仓库。
- 不会和正式答案目录 `algorithms/` 混在一起。

公开仓库只提交模板：

```text
templates/scratchpad/
├── README.md
├── current.py
├── reference.py
└── check.py
```

## 推荐用法

第一次使用时，创建本地草稿区：

```bash
mkdir -p scratchpad
cp templates/scratchpad/current.py scratchpad/current.py
cp templates/scratchpad/reference.py scratchpad/reference.py
cp templates/scratchpad/check.py scratchpad/check.py
```

日常练习流程：

```text
1. 在 scratchpad/current.py 里手敲自己的实现。
2. 运行 scratchpad/check.py，先看自己的 assert 是否通过。
3. 看正式答案后，把关键对照逻辑放到 scratchpad/reference.py。
4. 再运行 scratchpad/check.py，对比 current 和 reference 的输出。
5. 真正整理好的版本再写回 algorithms/ 对应题目目录。
```

## current.py 应该写什么

`current.py` 只放你当前正在手写的版本。

建议结构：

```python
def main():
    # 构造小数字张量
    # 写自己的函数
    # 打印关键 shape
    # 用 assert 检查结果
    return result
```

## reference.py 应该写什么

`reference.py` 放对照版本。

对照版本可以来自：

- 你自己第二次重写的版本。
- PyTorch 主实现与 NumPy 辅助版本的互相验证。
- `algorithms/` 里已经整理好的正式实现。

## check.py 做什么

`check.py` 会依次运行：

- `current.py` 的 `main()`
- `reference.py` 的 `main()`

如果两个 `main()` 都返回了结果，`check.py` 会尽量做 `allclose` 对比。更复杂的检查建议直接写在 `current.py` 或 `reference.py` 的 assert 里。
