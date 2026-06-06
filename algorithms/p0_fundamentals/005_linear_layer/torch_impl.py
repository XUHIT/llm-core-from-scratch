"""001 线性层：PyTorch 对照实现。

这个文件使用 PyTorch autograd 验证 NumPy 手写结果。
注意：nn.Linear 内部的 weight 形状是 [out_features, in_features]。
"""

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_toy_example() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造和 numpy_impl.py 完全相同的固定小数字样例。"""
    # x 有 2 个样本，每个样本 3 个输入特征
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    # NumPy 里的 W 形状是 [in_features, out_features]
    weight_math = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=torch.float64,
    )

    # PyTorch 的 nn.Linear 需要 [out_features, in_features]，所以这里转置
    weight_torch = weight_math.T.contiguous()
    bias = torch.tensor([0.7, 0.8], dtype=torch.float64)

    # 和 NumPy 文件保持同一个上游梯度
    grad_output = torch.tensor(
        [
            [1.0, -1.0],
            [0.5, 2.0],
        ],
        dtype=torch.float64,
    )
    return x, weight_torch, bias, grad_output


if __name__ == "__main__":
    x, weight_torch, bias, grad_output = build_toy_example()

    layer = Linear(3, 2).to(dtype=torch.float64)
    with torch.no_grad():
        # 手动写入固定权重，避免随机初始化影响学习和对照
        layer.linear.weight.copy_(weight_torch)
        layer.linear.bias.copy_(bias)

    y = layer(x)

    # backward 接收外部传入的 dL/dY，等价于 NumPy 里的 grad_output
    y.backward(grad_output)

    print("x.shape:", tuple(x.shape))
    print("weight.shape in nn.Linear:", tuple(layer.linear.weight.shape))
    print("bias.shape:", tuple(layer.linear.bias.shape))
    print("y.shape:", tuple(y.shape))
    print("\ny = linear(x)")
    print(y.detach().numpy())
    print("\ngrad_output = dL/dY")
    print(grad_output.numpy())
    print("\ngrad_x = dL/dX")
    print(x.grad.detach().numpy())
    print("\ngrad_weight_math = dL/dW")
    print(layer.linear.weight.grad.detach().numpy().T)
    print("\ngrad_bias = dL/db")
    print(layer.linear.bias.grad.detach().numpy())
