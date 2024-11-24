# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
import utils.transform.trans as trans

class Transform:
    def __init__(self) -> None:
        self._trans = {}
        self.add_trans()

    def add_trans(self):
        self.add_transform("zScore_std", trans.zScore_std, "标准化")
        self.add_transform("normalization_MinMax", trans.normalization_MinMax, "归一化")
        self.add_transform("multiple_zScore", trans.multiple_zScore, "n倍, z-score")
        self.add_transform("multiple_MinMax", trans.multiple_MinMax, "n倍, Min-Max归一化")
        self.add_transform("std_gaussianNoise", trans.std_gaussianNoise, "z-score, gaussian noise")
        self.add_transform("ewt_zScore", trans.ewt_zScore, "EWT, z-score")
        self.add_transform("dwt_zScore", trans.dwt_zScore, "DWT, z-score")
        self.add_transform("dwt_norml", trans.dwt_norml, "DWT, Min-Max")
        self.add_transform("dwtg_zScore", trans.dwtg_zScore, "DWTg, z-score")
        self.add_transform("fft_zScore", trans.fft_zScore, "fft, z-score")
        self.add_transform("fft", trans.fft, "FFT")
        self.add_transform("ewt", trans.ewt, "EWT")
        self.add_transform("dwt", trans.dwt, "DWT")


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

def get_transform(name):
    global __trans
    transform = __trans.get(name)
    return transform
