"""你当前手敲代码的地方。

这个文件复制到 scratchpad/current.py 后使用。
`scratchpad/` 不会被提交到 GitHub，可以放心改。
"""

from __future__ import annotations

import numpy as np


def main():
    """在这里写当前题目的最小验证。"""
    # 示例：先用 001_tensor_shape 的固定小张量练习。
    batch_size = 2
    seq_len = 3
    hidden_size = 4

    # x.shape = [B, T, D] = [2, 3, 4]
    x = np.arange(batch_size * seq_len * hidden_size, dtype=np.float64)
    x = x.reshape(batch_size, seq_len, hidden_size)

    # 把 [B, T, D] 展平成 [B*T, D]
    x_2d = x.reshape(batch_size * seq_len, hidden_size)

    print("x.shape:", x.shape)
    print("x_2d.shape:", x_2d.shape)

    assert x.shape == (2, 3, 4)
    assert x_2d.shape == (6, 4)

    # main() 可以返回你想和 reference.py 对比的结果。
    return x_2d


if __name__ == "__main__":
    main()
