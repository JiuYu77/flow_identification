# -*- coding: utf-8 -*-
import numpy as np
import random


class MinMaxNormalization:
    """
    线性归一化
    Min-Max 归一化（也称作离差标准化），即将特征xi的取值范围缩放到 [0,1] 区间内
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

