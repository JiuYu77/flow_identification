# -*- coding: UTF-8 -*-
from jyu.nn import Model, MODEL_YAML_DEFAULT

from jyu.nn.model import Model2, Model3

class YI(Model):
# class YI(Model2):
# class YI(Model3):
    """YI-Net"""
    def __init__(
            self,
            yaml_path=MODEL_YAML_DEFAULT,
            weights=None,
            scale:str=None,
            ch=1,
            verbose=True,
            fuse=False, split=False,
            initweightName='xavier',
            device='cpu'
    ):
        super().__init__(yaml_path, weights, scale, ch, verbose, fuse, split, initweightName, device)
        self.set_class_name()

    def set_names(self, names):
        self.names = names

    def set_class_name(self):
        '''修改模型名字'''
        self.__class__.__name__ = self.netName
