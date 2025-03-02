# -*- coding: UTF-8 -*-
import torch
from torch import nn

class ToVector(torch.nn.Module):
    def __init__(self):
        super(ToVector, self).__init__()
        self.avgPool = torch.nn.AdaptiveAvgPool1d(1)
        pass

    def __call__(self, x):
        batch, channel, _ = x.shape
        x = self.avgPool(x)
        x = x.view(batch, channel)
        return x

class ConvFusion(nn.Module):
    """
    通过卷积操作将两个数据进行融合
    """
    def __init__(self, in_channels1=512, in_channels2=512, out_length=512):
        super().__init__()
        self.in_length1 = in_channels1
        self.in_length2 = in_channels2
        self.out_length = out_length
        self.conv = nn.Conv1d(in_channels=in_channels1 + in_channels2, out_channels=out_length, kernel_size=3)
        self.bn = nn.BatchNorm1d(out_length)
        self.toVector = ToVector()

    def forward(self, feature1, feature2):
        feature = torch.cat((feature1, feature2), dim=1)
        feature = self.conv(feature)
        feature = self.toVector(feature)
        feature = self.bn(feature)
        return feature
