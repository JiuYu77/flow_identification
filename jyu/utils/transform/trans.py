# -*- coding: UTF-8 -*-
from .transforms import *

def zScore_std():
    """standardization标准化"""
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
