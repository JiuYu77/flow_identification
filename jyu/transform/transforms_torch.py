# -*- coding: utf-8 -*-
import torch
# import torchvision.transforms as transforms
# transforms.Compose()

class ToTensor:
    def __call__(self, x) -> torch.Tensor:
        """x是一个样本"""
        return torch.Tensor(x)
