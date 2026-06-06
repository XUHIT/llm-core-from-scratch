"""001 Linear Layer: PyTorch implementation.

y = xW^T + b
"""

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


if __name__ == "__main__":
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    layer = Linear(4, 3)
    y = layer(x)
    print(f"input:  {x.tolist()}")
    print(f"output: {y.tolist()}")
    print(f"shapes: {list(x.shape)} -> {list(y.shape)}")
