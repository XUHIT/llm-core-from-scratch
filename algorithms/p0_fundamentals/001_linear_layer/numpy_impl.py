"""001 Linear Layer: NumPy implementation.

This file uses the math convention y = xW + b, where W has shape
[in_features, out_features].
"""

from __future__ import annotations

import numpy as np


def linear_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Compute y = xW + b on the last dimension of x."""
    y = x @ weight
    if bias is not None:
        y = y + bias
    return y


def linear_backward(
    x: np.ndarray,
    weight: np.ndarray,
    grad_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gradients dx, dW, db for y = xW + b.

    Supports x with shape [..., in_features].
    """
    grad_x = grad_output @ weight.T

    x_2d = x.reshape(-1, x.shape[-1])
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

    grad_weight = x_2d.T @ grad_output_2d
    grad_bias = grad_output_2d.sum(axis=0)
    return grad_x, grad_weight, grad_bias


if __name__ == "__main__":
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    weight = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=np.float64,
    )
    bias = np.array([0.7, 0.8], dtype=np.float64)
    grad_output = np.ones((2, 2), dtype=np.float64)

    y = linear_forward(x, weight, bias)
    grad_x, grad_weight, grad_bias = linear_backward(x, weight, grad_output)

    print("x shape:", x.shape)
    print("weight shape:", weight.shape)
    print("bias shape:", bias.shape)
    print("y shape:", y.shape)
    print("y:")
    print(y)
    print("grad_x:")
    print(grad_x)
    print("grad_weight:")
    print(grad_weight)
    print("grad_bias:")
    print(grad_bias)
