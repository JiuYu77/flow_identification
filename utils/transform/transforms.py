# -*- coding: utf-8 -*-
import torchvision.transforms as transforms
import torch
import numpy as np
import random


class ToTensor:
    def __call__(self, x) -> torch.Tensor:
        """x是一个样本"""
        return torch.Tensor(x)

class MinMaxNormalization:
    """
    线性归一化
    Min-Max 标准化（也称作离差标准化），即将特征xi的取值范围缩放到 [0,1] 区间内
    """
    def __call__(self, x):
        min_ = np.min(x)
        max_ = np.max(x)
        x = (x - min_) / (max_ - min_)
        return x

class ZScoreStandardization:
    """z-score标准化"""
    def __init__(self) -> None:
        pass
    def __call__(self, x):
        mean = np.mean(x)
        std = np.nanstd(x)
        std = max(std, 0.001)
        x = (x - mean) / std
        return x

class Multiple:
    def __init__(self, n=100) -> None:
        self.n = n

    def __call__(self, x):
        return x * self.n

class randomSelector:
    """
    随机的选择一个transform分支
    """

    def __init__(self, transforms: list):
        super().__init__()
        self.transforms = transforms

    def __call__(self, x):
        trans = np.random.choice(self.transforms, 1)
        return trans(x)

class GaussianNoise:
    """高斯噪声"""
    def __init__(self, mean=0, std=0.05, prob=0.618) -> None:
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, x):
        p = random.uniform(0, 1)
        if p >= self.prob:
            noise = np.random.normal(self.mean, self.std, x.shape)
            x = x + noise
        return x

class SPNoise:
    """添加椒盐噪声"""
    def __call__(self, x):
        output = np.zeros(x.shape ,x.dtype)
        prob = random.uniform(0.0005,0.001)  #随机噪声比例
        thres = 1 - prob
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = x[i][j]
        return output
        # return x

class EWT:
    """Empirical Wavelet Transform"""
    def __init__(self, N=2) -> None:
        self.N = N

    def __call__(self, x):
        import ewtpy

        N = self.N
        x = x[0]
        ewt, mfb, boundaries = ewtpy.EWT1D(x, N=N)  # EWT
        ewt = ewt.reshape(1, -1)
        return ewt

class DWT:
    """Discrete Wavelet Transform"""
    def __init__(self, wavelet='db1') -> None:
        import pywt
        self.pywt = pywt

        self.wavelet = wavelet

    def __call__(self, x):
        # import pywt
        wavelet = self.wavelet

        # dwt
        pywt = self.pywt
        cA, cD = pywt.dwt(data=x, wavelet=wavelet)
        # print(cA.shape, cD.shape)
        out = np.concatenate((cA, cD), axis=1)  # concatenate 拼接
        # print(out.shape)
        return out

class DWTg:
    """Discrete Wavelet Transform"""
    def __init__(self, wavelet='db1') -> None:
        import pywt
        self.pywt = pywt

        self.wavelet = wavelet

    def __call__(self, x):
        # import pywt
        wavelet = self.wavelet

        #extend the signal by mirroring to deal with boundaries  不要镜像效果更好
        f = x
        ltemp = int(np.ceil(f.size/2)) #to behave the same as matlab's round
        fMirr =  np.append(np.flip(f[0:ltemp-1],axis = 0),f)  
        fMirr = np.append(fMirr,np.flip(f[-ltemp-1:-1],axis = 0))
        x = fMirr.reshape(1, -1)
        # print("x.shape: ", x.shape, x)
        x = np.fft.fft(x)
        # ff = abs(ff[0:int(np.ceil(ff.size/2))])#one-sided magnitude
        # print("xfft.shape: ", x.shape, x)
        x1 = x[:, :x.shape[1]//2]
        x2 = x[:, x.shape[1]//2:]

        # print("x.shape: ", x.shape, x)
        # print("x1.shape: ", x1.shape, x1)
        # print("x2.shape: ", x2.shape, x2)

        # dwt
        pywt = self.pywt
        cA, cD = pywt.dwt(data=x1, wavelet=wavelet)
        # print(cA.shape, cD.shape)
        out1 = np.concatenate((cA, cD), axis=1)  # concatenate 拼接

        cA, cD = pywt.dwt(data=x2, wavelet=wavelet)
        # print(cA.shape, cD.shape)
        out2 = np.concatenate((cA, cD), axis=1)  # concatenate 拼接
        # print("out2.shape", out2.shape)
        out = np.concatenate((out1, out2), axis=1)

        cA, cD = pywt.dwt(data=x, wavelet=wavelet)
        out = cD
        # print("out.shape", out.shape)
        return out

class FFT:
    """fast Fourier transform"""
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        x = np.fft.fft(x)
        out = abs(x)
        # print("out.shape", out.shape, out)
        return out

class FFT2:
    """fast Fourier transform"""
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        #extend the signal by mirroring to deal with boundaries  不要镜像效果更好
        # f = x
        # ltemp = int(np.ceil(f.size/2)) #to behave the same as matlab's round
        # fMirr =  np.append(np.flip(f[0:ltemp-1],axis = 0),f)  
        # fMirr = np.append(fMirr,np.flip(f[-ltemp-1:-1],axis = 0))
        # fMirr = fMirr.reshape(1, -1)
        # print("fMirr.shape: ", fMirr.shape, fMirr)
        # x = np.fft.fft(fMirr)

        # x = np.fft.fft(x[0])
        # print("x.shape: ", x.shape, x)
        # ff = abs(ff[0:int(np.ceil(ff.size/2))])#one-sided magnitude
        # x = abs(x[0:int(np.ceil(x.size/2))])  # one-sided magnitude
        # out = x.reshape(1, -1)
        # out = np.real(x).reshape(1, -1)
        x = np.fft.fft(x)
        out = abs(x)
        # print("out.shape", out.shape, out)
        return out

def standardization_zScore():
    """标准化"""
    return transforms.Compose([
        ZScoreStandardization(),
        ToTensor()
    ])

def multiple_zScore():
    """先变为n倍, 再z-score标准化"""
    return transforms.Compose([
        Multiple(),
        ZScoreStandardization(),
        ToTensor()
    ])

def normalization_MinMax():
    """归一化"""
    return transforms.Compose([
        MinMaxNormalization(),
        ToTensor()
    ])

def multiple_MinMax():
    """先变为n倍, 再Min-Max归一化"""
    return transforms.Compose([
        Multiple(),
        MinMaxNormalization(),
        ToTensor()
    ])

def std_gaussianNoise():
    return transforms.Compose([
        ZScoreStandardization(),
        GaussianNoise(),
        ToTensor()
    ])

def ewt_std():
    return transforms.Compose([
        EWT(N=2),
        ZScoreStandardization(),
        ToTensor()
    ])

def dwt_std():
    return transforms.Compose([
        DWT(),
        ZScoreStandardization(),
        ToTensor()
    ])

def dwtg_std():
    return transforms.Compose([
        DWTg(),
        ZScoreStandardization(),
        ToTensor()
    ])

def fft_std():
    return transforms.Compose([
        FFT(),
        ZScoreStandardization(),
        ToTensor()
    ])

def dwt_norml():
    return transforms.Compose([
        DWT(),
        MinMaxNormalization(),
        ToTensor()
    ])
