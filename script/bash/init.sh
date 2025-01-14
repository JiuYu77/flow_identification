#!/bin/bash
# encoding: UTF-8

# 0.安装Nvidia显卡驱动  Linux64
# 官网：https://www.nvidia.cn/

# 1.安装cuda和cudnn
# cuda: https://developer.nvidia.com/cuda-toolkit-archive
# CUDA 11.4
# cudnn: https://developer.nvidia.com/cudnn-downloads

# 2.安装PyTorch，官网：https://pytorch.org/get-started/locally/
# CUDA 11.3, torch+cu113 可以使用 cuda11.4
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 11.6
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# 3.安装其他Python包
pip install matplotlib==3.6.2 numpy==1.23.4 PyYAML==6.0 ewtpy==0.2 PyWavelets==1.4.1
