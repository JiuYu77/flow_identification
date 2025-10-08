# -*- coding: UTF-8 -*-
import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.append('.')

def calculate_mean_std(dataloader, channels=1):
    """计算数据集的均值和标准差"""
    channels = channels
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    total_samples = 0

    for X, _ in dataloader:
        # 计算当前batch的均值和方差
        batch_samples = X.size(0)
        X = X.view(batch_samples, X.size(1), -1)
        mean += X.mean(2).sum(0)
        std += X.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean.tolist(), std.tolist()

if __name__ == '__main__':
    from jyu.dataset.flowDataset import FlowDataset
    from jyu.dataloader.dataLoader_torch import data_loader
    datasetPath = "/root/my_flow/dataset/flow/v4/Pressure/4/train"
    sampleLength = 4096
    step = 2048
    transformName = "zScore_std"
    batchSize = 64
    shuffle=True
    dataloader =data_loader(FlowDataset, datasetPath, sampleLength, step, transformName, None, True, batchSize, shuffle, 4)

    mean, std = calculate_mean_std(dataloader)
    print("Mean:", mean)
    print("Std:", std)
