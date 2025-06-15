# -*- coding: UTF-8 -*-
from .transforms import *
from .Data_augmentation import *

def zScore_std():
    """standardization标准化"""
    return transforms.Compose([
        ZScoreStandardization(),
        ToTensor()
    ])

def normalization_MinMax():
    """归一化"""
    return transforms.Compose([
        MinMaxNormalization(),
        ToTensor()
    ])

def std_gaussianNoise():
    return transforms.Compose([
        ZScoreStandardization(),
        TSelector([ReturnData(), GaussianNoise()], [0.618, 0.3]),
        ToTensor()
    ])

def ewt():
    return transforms.Compose([
        EWT(N=2),
        ToTensor()
    ])

def ewt_zScore():
    return transforms.Compose([
        EWT(N=2),
        ZScoreStandardization(),
        ToTensor()
    ])

def dwt():
    return transforms.Compose([
        DWT(),
        ToTensor()
    ])

def dwt_zScore():
    return transforms.Compose([
        DWT(),
        ZScoreStandardization(),
        ToTensor()
    ])

def dwtg_zScore():
    return transforms.Compose([
        DWTg(),
        ZScoreStandardization(),
        ToTensor()
    ])

def dwt_norml():
    return transforms.Compose([
        DWT(),
        MinMaxNormalization(),
        ToTensor()
    ])

def fft():
    return transforms.Compose([
        FFT(),
        ToTensor()
    ])

def fft_zScore():
    return transforms.Compose([
        FFT(),
        ZScoreStandardization(),
        ToTensor()
    ])

def zScore_randomOne():
    """标准化 -> 随机选择 数据增强算法，ReturnData则不进行数据增强"""
    return transforms.Compose([
        ZScoreStandardization(),
        TRandomSelector([ReturnData(), GaussianNoise(), Reverse(), ScaleAmplitude(), TimeShift(), WindowWarp()]),
        ToTensor()
    ])

def zScore_probOne():
    """标准化 -> 概率选择 数据增强算法，ReturnData则不进行数据增强"""
    return transforms.Compose([
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
        ToTensor()
    ])
