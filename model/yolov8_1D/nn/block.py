# -*- coding: UTF-8 -*-
from torch import nn
from conv import Conv1d
import torch

class C2f1d(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5) -> None:
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.conv1 = Conv1d(c1, 2 * self.c, 1, 1)
        self.conv2 = Conv1d((2 + n) * self.c, c2, 1)
        # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(1, 1), e=1.0) for _ in range(n))
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f1d layer."""
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.conv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5) -> None:
        super().__init__()
        c_ = int(c2 * e) # hidden channels
        self.conv1 = Conv1d(c1, c_, k[0], 1)
        self.conv2 = Conv1d(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))
