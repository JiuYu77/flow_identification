# -*- coding:UTF-8 -*-
import torch
from torch import optim
import numpy as np
import sys
sys.path.append('.')

from tmp.beta3.beta4 import BetaVAE1D
from jyu.utils import data_loader, FlowDataset, tu

device = tu.get_device()
print(device)

# 定义模型和优化器
model = BetaVAE1D(1, latent_dim=3, beta=4, gamma=10.0, Capacity_max_iter=10000, loss_type='H')  # 可以调整beta值
optimizer = optim.Adam(model.parameters(), lr=1e-5)
model.to(device)

datasetPath = "/home/jyu/pro/my_git/dataset/flow/v4/Pressure/v4/train"
train_loader = data_loader(FlowDataset, datasetPath, 8192, 2048, "normalization_MinMax", supervised=False, batchSize=64, shuffle=True, numWorkers=4)
path = "tmp/beta3/"

# 训练过程
num_epochs = 100
batchNum = len(train_loader)
allSampleNum = len(train_loader.dataset)
minLoss = 1e6

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, _, index) in enumerate(train_loader):
        # print(data.shape)
        print("\033[Kbatch_idx: ", f"{batch_idx+1}/{batchNum}", end='\r')

        data = data.to(device)

        recon_batch, _, mu, logvar = model(data)
        loss_dict = model.loss_function(recon_batch, data, mu, logvar, M_N=model.beta)
        loss = loss_dict['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        sampleLoss = train_loss / allSampleNum

    print(f'Epoch {epoch+1}, Loss: {sampleLoss:.4f}')
    if minLoss > sampleLoss:
        minLoss = sampleLoss
        torch.save(model.state_dict(), path+"min_loss_params.pt")
torch.save(model.state_dict(), path+"last_params.pt")

