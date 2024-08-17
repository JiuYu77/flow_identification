# -*- coding: utf-8 -*-
from torch import nn

class n(nn.Module):
    def __init__(self, classNum) -> None:
        super().__init__()
        # net.apply(init_weight)
        self.apply(self.init_weight)

    def init_weight(m):
        """初始化参数"""
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

def nnn(classNum):
    return
