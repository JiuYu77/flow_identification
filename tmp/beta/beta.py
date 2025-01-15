# -*- coding: UTF-8 -*-
import torch
from torch import nn

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=20, beta=1.0):
        super(BetaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(64 * 7 * 7, 256),
            nn.Linear(64 * 32 * 32, 256),
            nn.ReLU()
        )

        # 输出均值和方差
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # 解码器部分
        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            # nn.Linear(256, 64 * 7 * 7),
            nn.Linear(256, 64 * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (64, -1)),
            # nn.Unflatten(1, (64, 32 * 32)),
            # nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1)
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        o = self.decoder(h)
        return torch.sigmoid(o)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # 重构误差 (Binary Cross-Entropy)
    # BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
    # BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 261952), x.view(-1, 262144), reduction='sum')
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 32*32), x.view(-1, 32*32), reduction='sum')

    # KL散度
    # KL散度 = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 其中，mu 和 logvar 是潜在空间的均值和对数方差
    # 注意：logvar是log(sigma^2)
    # 使用sum进行批次的加和
    # 标准正态分布p(z) = N(0, I)
    # q(z|x) = N(mu, sigma^2)
    # KL散度 = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    # 其中logvar是log(sigma^2)
    # 计算KL散度
    KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 最终损失函数
    return BCE + beta * KL_divergence
