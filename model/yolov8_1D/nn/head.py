# -*- coding: UTF-8 -*-
from torch import nn
from conv import Conv1d
import torch


class Classify(nn.Module):
    """YOLOv8_1D classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, prob=0.2) -> None:
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv1d(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=prob, inplace=True)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.drop(self.pool(self.conv(x)).flatten(1))
        x = self.linear(x)
        return x if self.training else x.softmax(1)
