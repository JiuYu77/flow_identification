# -*- coding: UTF-8 -*-
from torch import nn
import torch
from ..yolo.block import C2f1d
from ..yolo.transformer import TransformerBlock


class LSTMBranch(nn.Module):
    def __init__(self, input_size, hidden_size, output_features, num_layers=1):
        super(LSTMBranch, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_features)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])

        # _, (h_n, _) = self.lstm(x)  # 取最后一个时间步的隐藏状态
        # out = self.fc(h_n[-1, :, :])  # 使用最后一个时间步的隐藏状态
        return out

class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=batch_first,
                            bidirectional=bidirectional)  # bidirectional=True ==> 双向LSTM

    def get_hidden_state(self, x):
        '''初始化隐藏状态和细胞状态'''
        m = 1  # 单向LSTM
        if self.bidirectional:
            m = 2  # 双向LSTM
        h0 = torch.zeros(self.num_layers*m, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        c0 = torch.zeros(self.num_layers*m, x.size(0), self.hidden_size).to(x.device)  # 初始化细胞状态
        return h0, c0

    def forward(self, x):
        h0, c0 = self.get_hidden_state(x)
        x = x.view(x.size(0),1,-1)
        x, _ = self.lstm(x, (h0, c0))
        return x[:,-1,:]

    def _forward(self, x):
        x,_ = self.lstm(x)
        return x

class Gru(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=batch_first,
                            bidirectional=bidirectional)

    def get_hidden_state(self, x):
        '''初始化隐藏状态'''
        m = 1
        if self.bidirectional:
            m = 2
        h0 = torch.zeros(self.num_layers*m, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        c0 = torch.zeros(self.num_layers*m, x.size(0), self.hidden_size).to(x.device)  # 初始化细胞状态
        return h0, c0

    def forward(self, x):
        h0, c0 = self.get_hidden_state(x)
        x = x.view(x.size(0),1,-1)
        x, _ = self.gru(x, h0)
        return x[:,-1,:]

class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                            batch_first=batch_first,
                            bidirectional=bidirectional)  # bidirectional=True ==> 双向LSTM
    
    def get_hidden_state(self, x):
        '''初始化隐藏状态和细胞状态'''
        m = 1  # 单向RNN
        if self.bidirectional:
            m = 2  # 双向RNN
        h0 = torch.zeros(self.num_layers*m, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        c0 = torch.zeros(self.num_layers*m, x.size(0), self.hidden_size).to(x.device)  # 初始化细胞状态
        return h0, c0

    def forward(self, x):
        h0, c0 = self.get_hidden_state(x)
        x = x.view(x.size(0),1,-1)
        x, _ = self.rnn(x, h0)
        return x[:,-1,:]

class C2f1dTR(C2f1d):
    """Faster Implementation of CSP Bottleneck with 2 convolutions"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        # self.c = int(c2 * e)  # hidden channels
        # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(1, 1), e=1.0) for _ in range(n))
        # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        # self.m = TransformerBlock(c_, c_, 4, n)
        self.m = TransformerBlock(self.c, self.c, 4, n)

    def forward(self, x):
        """Forward pass through C2f1d layer."""
        y = list(self.conv1(x).chunk(2, 1))
        # y.extend(m(y[-1]) for m in self.m)
        y.extend(self.m(y[-1]))
        # y.extend(self.m(y[-1]).reshape(x.size(0), y[0].size(1), y[0].size(2)))
        return self.conv2(torch.cat(y, 1))

