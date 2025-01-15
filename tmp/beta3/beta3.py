# -*- coding: UTF-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class BetaVAE1D(nn.Module):
    def __init__(self, input_length, latent_dim=10, beta=1.0):
        super(BetaVAE1D, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # 编码器：输入 -> 潜在空间
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),  # 输出: (batch, 32, input_length//2)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # 输出: (batch, 64, input_length//4)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # 输出: (batch, 64, input_length//4)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # 输出: (batch, 128, input_length//8)
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            nn.Flatten()
        )

        # 计算编码后的长度
        conv_output_length = input_length // 8
        self.fc_mean = nn.Linear(128 * conv_output_length, latent_dim) # 潜在空间的均值
        self.fc_logvar = nn.Linear(128 * conv_output_length, latent_dim) # 潜在空间的对数方差

        # self.fc_mean = nn.Linear(256 * conv_output_length, latent_dim) # 潜在空间的均值
        # self.fc_logvar = nn.Linear(256 * conv_output_length, latent_dim) # 潜在空间的对数方差

        # 解码器：潜在空间 -> 重建数据
        self.decoder_input = nn.Linear(latent_dim, 128 * conv_output_length)
        self.decoder = nn.Sequential(
            # nn.ReLU(),
            nn.Unflatten(1, (256, conv_output_length)),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: (batch, 64, input_length//4)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: (batch, 64, input_length//4)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: (batch, 32, input_length//2)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出: (batch, 1, input_length)
            # nn.BatchNorm1d(1),
            # nn.AdaptiveAvgPool1d(4096),
            # nn.ReLU(),
            # nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1),  # 输出: (batch, 1, input_length)
            # nn.Tanh()
        )

        self.init_weight()

    def init_weight(self):
        for m in self.encoder:
            self.init_weight_xavier(m)
        for m in self.decoder:
            self.init_weight_xavier(m)
        self.init_weight_xavier(self.fc_mean)
        self.init_weight_xavier(self.fc_logvar)


    def init_weight_xavier(self, m):
        """初始化参数: 权重; xavier"""
        if type(m) == nn.Linear or type(m) == nn.Conv1d or type(m) == nn.ConvTranspose1d:
            nn.init.xavier_uniform_(m.weight)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # 重构误差 (MSE 或 MAE 可用于回归问题)
    batch_size = x.size(0)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum').div(batch_size)

    # KL 散度
    KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 最终损失
    return recon_loss + beta * KL_divergence
