"""对照实现的地方。

可以先留着这个最小版本。
看完正式答案后，也可以把关键对照逻辑放到这里。
"""

from __future__ import annotations

import torch


def main() -> torch.Tensor:
    """返回一个对照结果，用来和 current.py 比较。"""
    batch_size = 2
    seq_len = 3
    hidden_size = 4

    # 这里写得更直接一些，作为 current.py 的对照版本。
    result = torch.arange(batch_size * seq_len * hidden_size, dtype=torch.float64)
    result = result.reshape(batch_size * seq_len, hidden_size)

    print("reference.shape:", tuple(result.shape))
    assert result.shape == (6, 4)

    return result


if __name__ == "__main__":
    main()
