# -*- coding:UTF-8 -*-
import torch
from torch import optim

import sys
sys.path.append('.')

from tmp.beta2.Beta_VAE.model import BetaVAE_H, BetaVAE_B
# from tmp.beta2.Beta_VAE.solver import reconstruction_loss, kl_divergence
import torch.nn.functional as F

from jyu.utils import data_loader, FlowDataset, tu

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


# 定义模型和优化器
beta = 1.0
objective = 'H'
model = BetaVAE_H(z_dim=10, nc=1)  # 可以调整beta值
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
        recon_loss = reconstruction_loss(data, recon_batch, 'bernoulli')

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        if objective == 'H':
            beta_vae_loss = recon_loss + beta*total_kld
        elif objective == 'B':
            C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
            beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()

        beta_vae_loss.backward()
        train_loss += beta_vae_loss.item()

        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}')
