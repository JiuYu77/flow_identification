# -*- coding: UTF-8 -*-
from torch import nn
import torch
from pathlib import Path
from copy import deepcopy

import sys
sys.path.append('.')

from nn.modules import Conv1d, C2f1d
from nn.tasks import parse_model, yaml_model_load, guess_model_name, attempt_load_weights

from utils import LOGGER
from utils.torch_utils import InitWeight, fuse_conv_and_bn, de_parallel


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



# class Model(nn.Sequential):
class Model(nn.Module):
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
        super().__init__()
        self.names = None
        self.device = device
        self.fuse, self.split = fuse, split

        self.args = {"modelYaml": yaml_path, "fuse": fuse, "split": split}

        self.get_model(yaml_path, weights, ch, scale, verbose, device)

        if self.fuse:
            # self.fuse(self)
            self._fuse()
        if self.split:
            self._split()
        if weights is None:
            self.apply(InitWeight(initweightName).__call__)
        self.to(device)  # device: cpu, gpu(cuda)

    def get_model(self, yaml_path, weights, ch, scale, verbose, device):
        # yaml_ = yaml_model_load(yaml_path)
        # self.netName = guess_model_name(yaml_)

        # ch = yaml_["ch"] = yaml_.get("ch", ch)  # input channels
        # if scale:
        #     yaml_['scale'] = scale
        # self.scale, layers, save = parse_model(yaml_, ch, verbose=verbose)
        # self.add_layers(layers)

        if weights:  # 在已有权重上继续训练 or 测试 or 分类(推理, 实际使用)
            assert Path(weights).is_file(), f"{weights} does not exist."

            try:
                model = attempt_load_weights(weights, device)
                self.add_layers(model)

                self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                self.netName = model.netName
                self.scale = model.scale
                self.args = model.args
                self.fuse, self.split = model.fuse, model.split
            except:
                # **** 之后要删除 **** #
                yaml_ = yaml_model_load(yaml_path)
                self.netName = guess_model_name(yaml_)
                ch = yaml_["ch"] = yaml_.get("ch", ch)  # input channels
                if scale:
                    yaml_['scale'] = scale
                self.scale, layers, save = parse_model(yaml_, ch, verbose=verbose)
                self.add_layers(layers)
                model = torch.load(weights, map_location=device)
                self.load_state_dict(model)
                # **** 之后要删除 **** #
        else:
            yaml_ = yaml_model_load(yaml_path)
            self.netName = guess_model_name(yaml_)
            ch = yaml_["ch"] = yaml_.get("ch", ch)  # input channels
            if scale:
                yaml_['scale'] = scale
            self.scale, layers, save = parse_model(yaml_, ch, verbose=verbose)
            self.add_layers(layers)

    def add_layers(self, layers):
        for idx, module in enumerate(layers):
            self.add_module(str(idx), module)

    def _fuse(self):
        """
        fuse后效果不好
        """
        LOGGER.info('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, Conv1d) and hasattr(m, "bn1d"):
                m.conv1d = fuse_conv_and_bn(m.conv1d, m.bn1d)  # update conv; ValueError: can't optimize a non-leaf Tensor
                delattr(m, "bn1d")  # remove batchnorm
                m.forward = m.forward_fuse

    # def _fuse(self, model:nn.Module):
    #     for m in model.modules():
    #         if isinstance(m, Conv1d) and hasattr(m, "bn1d"):
    #             m.conv1d = fuse_conv_and_bn(m.conv1d, m.bn1d)  # update conv
    #             delattr(m, "bn1d")  # remove batchnorm
    #             m.forward = m.forward_fuse
    #     return model

    def _split(self):
        for m in self.modules():
            if isinstance(m, C2f1d):
                m.forward = m.forward_split

    def save(self, f):
        '''
        保存权重 params.pt
        f 保存路径, 如: ~/best_params.pt
        '''
        # state_dict = self.state_dict()
        # torch.save(state_dict, f)

        ckpt = {
            "model": deepcopy(de_parallel(self)).half(),
            "args": self.args,
        }
        torch.save(ckpt, f)
