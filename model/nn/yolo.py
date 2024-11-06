# -*- coding: UTF-8 -*-
from torch import nn
import torch
from pathlib import Path
import sys
sys.path.append('.')

from utils import LOGGER
from model.nn.modules import Conv1d, C2f1d

from model.nn.tasks import parse_model, yaml_model_load, guess_model_name
from utils.torch_utils import InitWeight, fuse_conv_and_bn

'''
scales: n s m l x
模型yaml配置文件 和 scales 拼在一起，作为参数传递给函数yolov8_1d()
yolov8_1D-cls.yaml  不拼接scales，默认使用规模n
yolov8_1Dn-cls.yaml
yolov8_1Ds-cls.yaml
yolov8_1Dm-cls.yaml
yolov8_1Dl-cls.yaml
yolov8_1Dx-cls.yaml
代码在 conf 文件夹，搜索配置文件
'''
MODEL_YAML_S = 'yolov8_1D/yolov8_1D-cls.yaml'
# MODEL_YAML_N = 'yolov8_1D/yolov8_1Dn-cls.yaml'
MODEL_YAML_DEFAULT = 'yolov8_1D-cls.yaml'
MODEL_YAML_N = 'yolov8_1Dn-cls.yaml'
MODEL_YAML_S = 'yolov8_1Ds-cls.yaml'
# MODEL_YAML = 'yolov8_1Dm-cls.yaml'
# MODEL_YAML = 'yolov8_1Dl-cls.yaml'
# MODEL_YAML = 'yolov8_1Dx-cls.yaml'



class Yolo(nn.Sequential):
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
    ):
        super().__init__()
        self.names = None
        self.get_model(yaml_path, weights, ch, scale, verbose, device)

        if fuse_:
            # self.fuse(self)
            self.fuse()
        if split_:
            self.split()
        if weights is None:
            self.apply(InitWeight(initweightName).__call__)
        self.to(device)  # device: cpu, gpu(cuda)

    def get_model(self, yaml_path, weights, ch, scale, verbose, device):
        yaml_ = yaml_model_load(yaml_path)
        self.netName = guess_model_name(yaml_)

        ch = yaml_["ch"] = yaml_.get("ch", ch)  # input channels
        if scale:
            yaml_['scale'] = scale
        self.scale, layers, save = parse_model(yaml_, ch, verbose=verbose)
        self.add_layers(layers)

        if weights:  # 在已有权重上继续训练 or 测试 or 分类(推理, 实际使用)
            assert Path(weights).is_file(), f"{weights} does not exist."
            self.model = torch.load(weights, map_location=device)
            self.load_state_dict(self.model)

    def add_layers(self, layers):
        for idx, module in enumerate(layers):
            self.add_module(str(idx), module)

    def fuse(self):
        """
        fuse后效果不好
        """
        LOGGER.info('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, Conv1d) and hasattr(m, "bn1d"):
                m.conv1d = fuse_conv_and_bn(m.conv1d, m.bn1d)  # update conv; ValueError: can't optimize a non-leaf Tensor
                delattr(m, "bn1d")  # remove batchnorm
                m.forward = m.forward_fuse

    # def fuse(self, model:nn.Module):
    #     for m in model.modules():
    #         if isinstance(m, Conv1d) and hasattr(m, "bn1d"):
    #             m.conv1d = fuse_conv_and_bn(m.conv1d, m.bn1d)  # update conv
    #             delattr(m, "bn1d")  # remove batchnorm
    #             m.forward = m.forward_fuse
    #     return model

    def split(self):
        for m in self.modules():
            if isinstance(m, C2f1d):
                m.forward = m.forward_split

    def save(self, f):
        '''
        保存权重 params.pt
        f 保存路径, 如: ~/best_params.pt
        '''
        state_dict = self.state_dict()
        torch.save(state_dict, f)


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

    def set_names(self, names):
        self.model.names = self.names = names

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
