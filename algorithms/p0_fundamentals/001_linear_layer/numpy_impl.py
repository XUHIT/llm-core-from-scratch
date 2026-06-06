"""001 线性层：NumPy 手写实现。

本文件使用数学里更直观的约定：

    y = x @ W + b

其中 W 的形状是 [in_features, out_features]。
"""

from __future__ import annotations

import numpy as np


def linear_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """计算线性层前向传播。

    参数：
        x: 输入张量，形状为 [..., in_features]
        weight: 权重矩阵，形状为 [in_features, out_features]
        bias: 偏置向量，形状为 [out_features]

    返回：
        输出张量，形状为 [..., out_features]
    """
    # 矩阵乘法只作用在最后一维：[..., in_features] @ [in_features, out_features]
    y = x @ weight
    if bias is not None:
        # bias 会自动广播到 x 前面的 batch/sequence 维度上
        y = y + bias
    return y


def linear_backward(
    x: np.ndarray,
    weight: np.ndarray,
    grad_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算线性层反向传播。

    已知：
        y = x @ W + b
        grad_output = dL/dy

    返回：
        grad_x: dL/dx
        grad_weight: dL/dW
        grad_bias: dL/db
    """
    # dX = dY @ W.T，形状 [..., out_features] @ [out_features, in_features]
    grad_x = grad_output @ weight.T

    # dW 和 db 需要对所有非特征维度求和。
    # 这里把 x 展平成 [N, in_features]，把 dY 展平成 [N, out_features]。
    x_2d = x.reshape(-1, x.shape[-1])
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

    # dW = X.T @ dY，形状 [in_features, N] @ [N, out_features]
    grad_weight = x_2d.T @ grad_output_2d

    # db = sum(dY)，对 batch/sequence 维度求和，只保留 out_features 维度
    grad_bias = grad_output_2d.sum(axis=0)
    return grad_x, grad_weight, grad_bias


def build_toy_example() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造固定小数字样例，方便手算和对照 PyTorch。"""
    # x 有 2 个样本，每个样本 3 个输入特征
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )

    # W 把 3 维输入映射到 2 维输出
    weight = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=np.float64,
    )

    # b 会广播到每一个样本上
    bias = np.array([0.7, 0.8], dtype=np.float64)

    # 假设上游梯度 dL/dY 已知。这里不用全 1，是为了更容易发现矩阵方向写错的问题。
    grad_output = np.array(
        [
            [1.0, -1.0],
            [0.5, 2.0],
        ],
        dtype=np.float64,
    )
    return x, weight, bias, grad_output


if __name__ == "__main__":
    x, weight, bias, grad_output = build_toy_example()
    y = linear_forward(x, weight, bias)
    grad_x, grad_weight, grad_bias = linear_backward(x, weight, grad_output)

    print("x.shape:", x.shape)
    print("weight.shape:", weight.shape)
    print("bias.shape:", bias.shape)
    print("y.shape:", y.shape)
    print("\ny = x @ W + b")
    print(y)
    print("\ngrad_output = dL/dY")
    print(grad_output)
    print("\ngrad_x = dL/dX")
    print(grad_x)
    print("\ngrad_weight = dL/dW")
    print(grad_weight)
    print("\ngrad_bias = dL/db")
    print(grad_bias)
