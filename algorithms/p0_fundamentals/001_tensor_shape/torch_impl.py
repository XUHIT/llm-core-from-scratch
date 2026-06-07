"""001 Tensor Shape：PyTorch 维度变化与广播示例。

本文件与 numpy_impl.py 使用相同的 shape 设计。
额外强调 PyTorch 中 transpose 后的 contiguous 问题。
"""

from __future__ import annotations

import torch


def make_input(batch_size: int = 2, seq_len: int = 3, hidden_size: int = 4) -> torch.Tensor:
    """构造一个容易观察元素顺序的小张量。

    返回：
        x: 形状为 [B, T, D]
    """
    values = torch.arange(batch_size * seq_len * hidden_size, dtype=torch.float64)
    return values.reshape(batch_size, seq_len, hidden_size)


def flatten_batch_time(x: torch.Tensor) -> torch.Tensor:
    """把 [B, T, D] 展平成 [B*T, D]。"""
    assert x.ndim == 3
    batch_size, seq_len, hidden_size = x.shape
    return x.reshape(batch_size * seq_len, hidden_size)


def restore_batch_time(x_2d: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """把 [B*T, D] 恢复成 [B, T, D]。"""
    assert x_2d.ndim == 2
    hidden_size = x_2d.shape[-1]
    return x_2d.reshape(batch_size, seq_len, hidden_size)


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """把 [B, T, D] 拆成 [B, H, T, Dh]。"""
    assert x.ndim == 3
    batch_size, seq_len, hidden_size = x.shape
    assert hidden_size % num_heads == 0

    head_dim = hidden_size // num_heads

    # 先拆最后一维：[B, T, D] -> [B, T, H, Dh]
    x = x.reshape(batch_size, seq_len, num_heads, head_dim)

    # 再把 head 维度放到 time 前面：[B, T, H, Dh] -> [B, H, T, Dh]
    return x.transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """把 [B, H, T, Dh] 合并回 [B, T, D]。"""
    assert x.ndim == 4
    batch_size, num_heads, seq_len, head_dim = x.shape

    # transpose 后通常是 non-contiguous，先 contiguous 再 view 更直观
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, seq_len, num_heads * head_dim)


def add_feature_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """演示 [D] 如何广播到 [B, T, D]。"""
    assert x.ndim == 3
    assert bias.shape == (x.shape[-1],)
    return x + bias


def add_position_embedding(x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
    """演示 [T, D] 如何广播到 [B, T, D]。"""
    assert x.ndim == 3
    assert pos_emb.shape == (x.shape[1], x.shape[2])

    # pos_emb 前面缺少 batch 维，会自动广播成 [B, T, D]
    return x + pos_emb


def apply_causal_mask(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """给 attention scores 加 causal mask。

    参数：
        scores: 形状为 [B, H, T, T]

    返回：
        masked_scores: 被 mask 的位置填成 -1e9
        mask: 形状为 [T, T]，True 表示当前位置不可见
    """
    assert scores.ndim == 4
    batch_size, num_heads, query_len, key_len = scores.shape
    assert query_len == key_len

    del batch_size, num_heads

    # 上三角位置代表“看未来 token”，需要被 mask 掉
    mask = torch.triu(torch.ones(query_len, key_len, dtype=torch.bool), diagonal=1)

    # mask[None, None, :, :] 会广播到 [B, H, T, T]
    masked_scores = scores.masked_fill(mask[None, None, :, :], -1e9)
    return masked_scores, mask


def run_shape_walkthrough() -> dict[str, torch.Tensor]:
    """执行完整 shape 走查，并返回中间结果。"""
    batch_size = 2
    seq_len = 3
    hidden_size = 4
    num_heads = 2

    x = make_input(batch_size, seq_len, hidden_size)
    x_2d = flatten_batch_time(x)
    restored = restore_batch_time(x_2d, batch_size, seq_len)

    heads = split_heads(x, num_heads)
    merged = merge_heads(heads)

    bias = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    x_with_bias = add_feature_bias(x, bias)

    pos_emb = torch.arange(seq_len * hidden_size, dtype=torch.float64).reshape(seq_len, hidden_size) / 100.0
    x_with_pos = add_position_embedding(x, pos_emb)

    scores = torch.arange(batch_size * num_heads * seq_len * seq_len, dtype=torch.float64)
    scores = scores.reshape(batch_size, num_heads, seq_len, seq_len)
    masked_scores, mask = apply_causal_mask(scores)

    assert x.shape == (2, 3, 4)
    assert x_2d.shape == (6, 4)
    assert restored.shape == x.shape
    assert heads.shape == (2, 2, 3, 2)
    assert merged.shape == x.shape
    assert x_with_bias.shape == x.shape
    assert x_with_pos.shape == x.shape
    assert scores.shape == (2, 2, 3, 3)
    assert mask.shape == (3, 3)
    assert masked_scores.shape == scores.shape

    # 展平再恢复、拆 head 再合并，都不应该改变元素顺序
    assert torch.equal(restored, x)
    assert torch.equal(merged, x)

    return {
        "x": x,
        "x_2d": x_2d,
        "heads": heads,
        "merged": merged,
        "x_with_bias": x_with_bias,
        "x_with_pos": x_with_pos,
        "scores": scores,
        "mask": mask,
        "masked_scores": masked_scores,
    }


if __name__ == "__main__":
    outputs = run_shape_walkthrough()

    for name, value in outputs.items():
        print(f"{name}.shape: {tuple(value.shape)}")

    print("\nx[0] =")
    print(outputs["x"][0])

    print("\nheads[0, 0]：第 0 个样本、第 0 个 head")
    print(outputs["heads"][0, 0])

    print("\ncausal mask：True 表示不能看未来位置")
    print(outputs["mask"])

    print("\n所有 PyTorch shape 检查通过。")
