# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
import utils.transform.transforms as transforms

class Transform:
    def __init__(self) -> None:
        self._trans = {}
        self.add_trans()
    
    def add_trans(self):
        self.add_transform("standardization_zScore", transforms.standardization_zScore, "标准化")
        self.add_transform("normalization_MinMax", transforms.normalization_MinMax, "归一化")
        self.add_transform("multiple_zScore", transforms.multiple_zScore, "n倍, z-score")
        self.add_transform("multiple_MinMax", transforms.multiple_MinMax, "n倍, Min-Max归一化")
        self.add_transform("std_gaussianNoise", transforms.std_gaussianNoise, "z-score, gaussian noise")
        self.add_transform("ewt_std", transforms.ewt_std, "EWT, z-score")
        self.add_transform("dwt_std", transforms.dwt_std, "DWT, z-score")
        self.add_transform("dwt_norml", transforms.dwt_norml, "DWT, Min-Max")
        self.add_transform("dwtg_std", transforms.dwtg_std, "DWTg, z-score")
        self.add_transform("fft_std", transforms.fft_std, "fft, z-score")
        self.add_transform("fft", transforms.fft, "FFT")
        self.add_transform("ewt", transforms.ewt, "EWT")
        self.add_transform("dwt", transforms.dwt, "DWT")


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
