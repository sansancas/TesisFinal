from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


class SpatialDropout1d(nn.Module):
    """Applies the same dropout mask across the temporal axis."""

    def __init__(self, p: float) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        # x: (B, T, C)
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)
        return x * mask


class SEBlock1d(nn.Module):
    def __init__(self, channels: int, se_ratio: int = 16) -> None:
        super().__init__()
        reduced = max(1, channels // se_ratio)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        z = self.pool(x.transpose(1, 2)).squeeze(-1)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        z = z.unsqueeze(1)
        return x * z


class FiLM1d(nn.Module):
    def __init__(self, channels: int, feat_dim: int | None = None) -> None:
        super().__init__()
        feat_dim = int(feat_dim or channels)
        self.g = nn.Linear(feat_dim, channels)
        self.b = nn.Linear(feat_dim, channels)

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        gamma = self.g(feat).unsqueeze(1)
        beta = self.b(feat).unsqueeze(1)
        return gamma * x + beta


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        separable: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = (kernel_size - 1) * dilation
        if separable:
            self.depthwise = nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels,
                dilation=dilation,
                padding=0,
                bias=bias,
            )
            self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        else:
            self.depthwise = None
            self.pointwise = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=0,
                bias=bias,
            )
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expects shape (B, C, T)
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        if self.depthwise is not None:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.pointwise(x)
        return x


class AttentionPooling1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        w = self.score(x)
        w = torch.softmax(w, dim=1)
        return torch.sum(w * x, dim=1)