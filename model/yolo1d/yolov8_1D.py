# -*- coding: UTF-8 -*-

from model.nn.yolo import *

class Yolov8_1D(Yolo1d):
    def __init__(
            self,
            yaml_path=MODEL_YAML_DEFAULT,
            weights=None,
            scale:str=None,
            ch=1,
            verbose=True,
            fuse_=False, split_=False,
            initweightName='xavier',
            device='cpu'
    ) -> None:
        super().__init__(yaml_path,  weights, scale, ch, verbose, fuse_, split_, initweightName, device)