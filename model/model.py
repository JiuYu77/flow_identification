# -*- coding: UTF-8 -*-
from nn import Yolo, MODEL_YAML_DEFAULT

class YOLO1D:
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
        self.model = Yolo(yaml_path,  weights, scale, ch, verbose, fuse_, split_, initweightName, device)
        self.scale = self.model.scale
        self.fuse, self.split, self.initweightName = fuse_, split_, initweightName

        self.__class__.__name__ = self.model.netName  # 修改模型名字
        self.names = self.model.names

    def set_names(self, names):
        self.names = names
        self.model.names = names

    def __call__(self, X, *args, **kwargs):
        return self.model(X)

    def eval(self):
        '''评估模式'''
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def modules(self):
        return self.model.modules()

    def train(self):
        '''训练模式'''
        self.model.train()

    def state_dict(self):
        return self.model.state_dict()

    def save(self, f):
        return self.model.save(f)
