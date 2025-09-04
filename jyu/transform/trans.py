# -*- coding: UTF-8 -*-
from .transforms import *
from .Data_augmentation import *


class Compose:
    """组合各种转换操作"""
    def __init__(self, transforms:list):
        self.transforms = transforms

    def __call__(self, x, *args, **kwds):
        for t in self.transforms:
            x = t(x)
        return x

    def append(self, transform):
        self.transforms.append(transform)

def zScore_std():
    """standardization标准化"""
    return Compose([
        ZScoreStandardization()
    ])

def MinMax_normalization():
    """归一化"""
    return Compose([
        MinMaxNormalization(),
    ])

def std_gaussianNoise():
    return Compose([
        ZScoreStandardization(),
        TSelector([ReturnData(), GaussianNoise()], [0.618, 0.3]),
    ])

def ewt():
    return Compose([
        EWT(N=2),
    ])

def ewt_zScore():
    return Compose([
        EWT(N=2),
        ZScoreStandardization(),
    ])

def dwt():
    return Compose([
        DWT(),
    ])

def dwt_zScore():
    return Compose([
        DWT(),
        ZScoreStandardization(),
    ])

def dwtg_zScore():
    return Compose([
        DWTg(),
        ZScoreStandardization(),
    ])

def dwt_norml():
    return Compose([
        DWT(),
        MinMaxNormalization(),
    ])

def fft():
    return Compose([
        FFT(),
    ])

def fft_zScore():
    return Compose([
        FFT(),
        ZScoreStandardization(),
    ])

def zScore_randomOne():
    """标准化 -> 随机选择 数据增强算法，ReturnData则不进行数据增强"""
    return Compose([
        ZScoreStandardization(),
        TRandomSelector([ReturnData(), GaussianNoise(), Reverse(), ScaleAmplitude(), TimeShift(), WindowWarp()]),
    ])

def zScore_probOne():
    """标准化 -> 概率选择 数据增强算法，ReturnData则不进行数据增强"""
    return Compose([
        ZScoreStandardization(),
        # TSelector([ReturnData(), GaussianNoise(), Reverse(), ScaleAmplitude(), TimeShift(), WindowWarp(), SliceSplice(), FrequencyPerturb()],
        #           [0.6, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]),
        # TSelector([ReturnData(), GaussianNoise(), Reverse(), ScaleAmplitude(), TimeShift(), WindowWarp()],
        #           [0.6, 0.08, 0.08, 0.08, 0.08, 0.08]),
        # TSelector([ReturnData(), GaussianNoise(), Reverse(), ScaleAmplitude(), TimeShift(), SliceSplice(), FrequencyPerturb()],
        #           [0.6, 0.066, 0.066, 0.066, 0.066, 0.066, 0.066]),
        # TSelector([ReturnData(), GaussianNoise(), Reverse(), ScaleAmplitude(), WindowWarp(), SliceSplice(), FrequencyPerturb()],
        #           [0.6, 0.066, 0.066, 0.066, 0.066, 0.066, 0.066]),
        # TSelector([ReturnData(), GaussianNoise(), Reverse(), TimeShift(), WindowWarp(), SliceSplice(), FrequencyPerturb()],
        #           [0.6, 0.066, 0.066, 0.066, 0.066, 0.066, 0.066]),
        # TSelector([ReturnData(), GaussianNoise(), ScaleAmplitude(), TimeShift(), SliceSplice()],
        #           [0.6, 0.1, 0.1, 0.1, 0.1]),
        TSelector([ReturnData(), GaussianNoise(), ScaleAmplitude(), TimeShift(), WindowWarp(), SliceSplice(), FrequencyPerturb()],
                  [0.6, 0.066, 0.066, 0.066, 0.066, 0.066, 0.066]),
    ])
