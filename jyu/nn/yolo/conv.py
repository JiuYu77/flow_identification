# -*- coding: UTF-8 -*-
from torch import nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv1d(nn.Module):
    """一维卷积，批规范化"""

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, act=True) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), dilation=d, groups=g, bias=False)  # 一维卷积kernel_size是k，而不是 1xk
        self.bn1d = nn.BatchNorm1d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn1d(self.conv1d(x)))

    def forward_fuse(self, x):
        return self.act(self.conv1d(x))
