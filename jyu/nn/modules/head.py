# -*- coding: UTF-8 -*-
from torch import nn
import torch

from ..yolo.conv import Conv1d
from .block import Lstm, Gru, Rnn


class ClassifyV2_(nn.Module):
    """YI-Netv2 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

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
        return x

class ClassifyV22(nn.Module):
    """YI-Netv2 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, prob=0.2) -> None:
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv1d(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=prob, inplace=True)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.drop(self.pool(self.conv(x)).flatten(1))
        return x

class ClassifyV2(nn.Module):
    """YI-Netv2 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, prob=0.2) -> None:
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv1d(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=prob, inplace=True)
        # c3 = 128
        # c3 = 256
        c3 = 512
        # c3 = 1024
        self.lstm = Lstm(c_, c3, 1)
        # self.lstm = LSTM(c_, c3, 2)
        self.linear = nn.Linear(c3*2, c2) # c3  c3*2

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.drop(self.pool(self.conv(x)).flatten(1))

        x = self.lstm(x)
        x = self.linear(x)
        return x

class ClassifyV24(nn.Module):
    """YI-Netv2 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, prob=0.2) -> None:
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv1d(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=prob, inplace=True)
        # c3 = 128
        # c3 = 256
        c3 = 512
        # c3 = 1024
        self.gru = Gru(c_, c3, 1)
        # self.gru = Gru(c_, c3, 2)
        self.linear = nn.Linear(c3*2, c2) # c3  c3*2

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.drop(self.pool(self.conv(x)).flatten(1))

        x = self.gru(x)
        x = self.linear(x)
        return x

class ClassifyV25(nn.Module):
    """YI-Netv2 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, prob=0.2) -> None:
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv1d(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=prob, inplace=True)
        c3 = 256
        # c3 = 512
        # c3 = 1024
        self.rnn = Rnn(c_, c3, 1)
        # self.rnn = Rnn(c_, c3, 2)
        self.linear = nn.Linear(c3*2, c2) # c3  c3*2

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.drop(self.pool(self.conv(x)).flatten(1))

        x = self.rnn(x)
        x = self.linear(x)
        return x

class ClassifyV26(nn.Module):
    """YI-Netv2 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, prob=0.2) -> None:
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv1d(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=prob, inplace=True)
        c3 = 128
        # c3 = 256
        # c3 = 512
        # c3 = 1024
        self.lstm = Lstm(c_, c3, 1)
        c4 = 128
        # c4 = 256
        # c4 = 512
        # c4 = 1024
        self.gru = Gru(c_, c4, 1)
        # self.rnn = Rnn(c_, c3, 2)
        self.linear = nn.Linear(c3*2 + c4*2, c2) # c3  c3*2
        # self.linear0 = nn.Conv1d(1,1, kernel_size=3, stride=2, padding=1) # c3  c3*2
        # self.bn = nn.BatchNorm1d((c3*2 + c4*2)//2)
        # self.linear = nn.Linear((c3*2 + c4*2)//2, c2) # c3  c3*2

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.drop(self.pool(self.conv(x)).flatten(1))

        o1 = self.lstm(x)
        o2 = self.gru(x)
        x = torch.cat((o1, o2), dim=1)
        # x = x.view(x.size(0), 1, -1)
        # x = self.linear0(x)
        # x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class Classifier(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.linear = nn.Linear(c1, c2)

    def forward(self, x):
        x = self.linear(x)
        return x if self.training else x.softmax(1)
