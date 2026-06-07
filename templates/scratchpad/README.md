# Scratchpad Template

这是本地草稿区模板。默认示例使用 PyTorch。

复制到 `scratchpad/` 后使用：

```bash
mkdir -p scratchpad
cp templates/scratchpad/current.py scratchpad/current.py
cp templates/scratchpad/reference.py scratchpad/reference.py
cp templates/scratchpad/check.py scratchpad/check.py
python scratchpad/check.py
```

文件分工：

```text
current.py    # 你当前手敲的实现
reference.py  # 对照实现，可以先留空
check.py      # 快速运行 current/reference
```

练习时优先在 `current.py` 里写：

- 小数字 PyTorch Tensor 输入。
- shape 打印。
- 自己手写的函数。
- assert 检查。

整理完成后，再把稳定版本迁移到 `algorithms/` 的对应题目目录。
