# -*- coding: UTF-8 -*-
from torch import nn

class SENet1d(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SENet1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)  # 序列数据
        return x * y
        # return x * y.expand_as(x)

    def forward_bak(self, x):
        '''备份，原本处理图像的SENet使用'''
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)  # RGB图像
        return x * y
        # return x * y.expand_as(x)
