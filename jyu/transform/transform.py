# -*- coding: utf-8 -*-
"""本文件是调用接口，需import"""
import sys
sys.path.append('.')
import jyu.transform.trans as trans


trainTransformList = [
    "zScore_std",
    "MinMax_norm",
    "std_gaussianNoise",
    "ewt_zScore",
    "dwt_zScore",
    "dwtg_zScore",
    "fft_zScore",
    "zScore_randomOne",
    "zScore_probOne"
]

testTransformList = [
    "zScore_std",
    "MinMax_norm",
    "ewt_zScore",
    "dwt_zScore",
    "dwtg_zScore",
    "fft_zScore",
]

class Transform:
    def __init__(self) -> None:
        self._trans = {}
        self.add_trans()

    def add_trans(self):
        self.add_transform("zScore_std", trans.zScore_std, "标准化")
        self.add_transform("MinMax_norm", trans.MinMax_normalization, "归一化")
        self.add_transform("std_gaussianNoise", trans.std_gaussianNoise, "z-score, gaussian noise")
        self.add_transform("ewt_zScore", trans.ewt_zScore, "EWT, z-score")
        self.add_transform("dwt_zScore", trans.dwt_zScore, "DWT, z-score")
        self.add_transform("dwt_norml", trans.dwt_norml, "DWT, Min-Max")
        self.add_transform("dwtg_zScore", trans.dwtg_zScore, "DWTg, z-score")
        self.add_transform("fft_zScore", trans.fft_zScore, "fft, z-score")
        self.add_transform("fft", trans.fft, "FFT")
        self.add_transform("ewt", trans.ewt, "EWT")
        self.add_transform("dwt", trans.dwt, "DWT")
        self.add_transform("zScore_randomOne", trans.zScore_randomOne, "随机选择一个转换操作")
        self.add_transform("zScore_probOne", trans.zScore_probOne, "按概率选择一个转换操作")


    def add_transform(self, name, transform, desc):
        self._trans[name] = {
            "transform": transform,
            "desc": desc
        }

    def get(self, name):
        # assert name in self._trans, f"{name} transform not exist"
        if name not in self._trans:
            raise KeyError(f"{name} transform not exist")
        return self._trans[name]['transform']()

__trans = Transform()

def get_transform(name, toTensor:bool=True):
    global __trans
    transform = __trans.get(name)
    if toTensor:
        from .transforms_torch import ToTensor
        transform.append(ToTensor())
    return transform
