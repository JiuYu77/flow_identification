# -*- coding: UTF-8 -*-
from jyu.nn import Model, MODEL_YAML_DEFAULT

class YOLO1D:
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
    ) -> None:
        self.model = Model(yaml_path,  weights, scale, ch, verbose, fuse, split, initweightName, device)
        self.scale = self.model.scale
        self.fuse, self.split, self.initweightName = fuse, split, initweightName

        self.__class__.__name__ = self.model.netName  # 修改模型名字
        self.names = self.model.names

    def set_names(self, names):
        '''设置类别'''
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

class YOLOv8_1D(YOLO1D):
    pass

class YOLOv11_1D(YOLO1D):
    pass
