# -*- coding:UTF-8 -*-
import torch
from torch import optim

import sys
sys.path.append('.')

from tmp.beta.beta import BetaVAE, loss_function
from jyu.utils import data_loader, FlowDataset, tu

# 定义模型和优化器
model = BetaVAE(latent_dim=20, beta=4.0)  # 可以调整beta值
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = tu.get_device()
print(device)

datasetPath = "/home/jyu/apro/flow/dataset/flow/v4/Pressure/v4/train"
train_loader = data_loader(FlowDataset, datasetPath, 4096, 2048, "zScore_std", batchSize=64, shuffle=True, numWorkers=4)

# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        print(data.shape)
        print("\033[Kbatch_idx: ", batch_idx)
        # data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, beta=4.0)
        loss.backward()
        train_loss += loss.item()

        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}')
