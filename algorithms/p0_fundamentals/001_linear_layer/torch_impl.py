"""001 Linear Layer: PyTorch reference implementation."""

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


if __name__ == "__main__":
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float64,
    )
    layer = Linear(3, 2).to(dtype=torch.float64)
    with torch.no_grad():
        layer.linear.weight.copy_(
            torch.tensor(
                [
                    [0.1, 0.3, 0.5],
                    [0.2, 0.4, 0.6],
                ],
                dtype=torch.float64,
            )
        )
        layer.linear.bias.copy_(torch.tensor([0.7, 0.8], dtype=torch.float64))

    y = layer(x)
    print(f"input:  {x.tolist()}")
    print(f"output: {y.tolist()}")
    print(f"shapes: {list(x.shape)} -> {list(y.shape)}")
