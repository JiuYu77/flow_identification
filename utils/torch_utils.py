# -*- coding: utf-8 -*-
import torch
import os
import sys
import time
import numpy as np
from torch import nn, optim
from .lion import Lion
from .__init__ import LOGGER, colorstr


def get_device(i=0):
    """如果gpu可用，则返回gpu，否则返回cpu"""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")

def getDeviceName():
    """
    :brief 获取当前设备的名称
    :return 设备名
        - hy    恒源云
        - windows
        - linux
        - mac   lyn 的 Macbook Pro M2
    """
    if os.path.isdir("/hy-tmp"):  # 有/hy-tmp目录，则在是恒源云服务器上执行程序
        return "hy"
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    return "mac"


class InitWeight:
    def __init__(self, name='xavier') -> None:
        func = f'self.init_weight_{name}'
        self.__call__ = eval(func)

    def __call__(self):
        pass

    def init_weight_xavier(self, m):
        """初始化参数: 权重; xavier"""
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)

    def init_weight_kaiming(self, m):
        """kaiming"""
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 权重初始化

def smart_optimizer(model:nn.Module, name, lr, momentum=0.5, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm1d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if not p.is_leaf:  # 如果fuse()了，会出现非叶子张量，则对非叶张量进行detach，防止报错
                p.detach_()
            if p_name == 'bias':  # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == 'SGD':
        optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=False)
    elif name == 'Adam':
        optimizer = optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'LION':
        optimizer = Lion(g[2], lr=lr, betas=(momentum, 0.99), weight_decay=0.0)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm1d weights)
    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    LOGGER.info(f"\n{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias\n')
    return optimizer

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """n个变量分别累加"""
    def __init__(self, n) -> None:
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def val(net:torch.nn.Module, valIter, device, loss, epoch_, epochNum, resultPath):
    """验证"""
    valIterPath = os.path.join(resultPath, 'val_iter')
    net.eval()
    accumulatorVal = Accumulator(3)
    batchNum = len(valIter)
    for i, (X, y) in enumerate(valIter):
        # print(X[len(X)-1], f"   {len(X)}   ", f"\t\033[31m{i}\033[0m   {type(X)}")
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss_ = loss(y_hat, y)
        correctNum = accuracy(y_hat, y)
        sampleNum = X.shape[0]
        val_loss = loss_.item()
        batchLoss = val_loss * sampleNum
        accumulatorVal.add(batchLoss, correctNum, sampleNum)
        prg = f"{int((i + 1) / batchNum * 100)}%"  # 进度，百分比
        print(f"\r\033[K\033[31mepoch\033[0m {epoch_:>3}/{epochNum}    \033[31mbatch:\033[0m{i}    \033[31mprogress:\033[0m{prg}    \033[31msample_num:\033[0m{X.shape[0]}    \033[31mval_loss:\033[0m{val_loss:.5f}", end="\r")
        if prg == '100%':
            time.sleep(0.1)

        # 训练数据记录
        batchAcc = correctNum / sampleNum
        with open(valIterPath, 'a+') as tIter_fp:
            tIter_fp.write(
                f"{epoch_:>6}\t{i+1:>6}\t{sampleNum:>6}\t{correctNum:>6}\t{batchAcc:>6.4}\t{batchLoss:>6.2f}\t{val_loss:>6.3f}\n")

    valLoss = accumulatorVal[0] / accumulatorVal[2]
    valAcc = accumulatorVal[1] / accumulatorVal[2]
    return valLoss, valAcc
