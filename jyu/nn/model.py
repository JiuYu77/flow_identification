# -*- coding: UTF-8 -*-
from torch import nn
import torch
from pathlib import Path
from copy import deepcopy

import sys
sys.path.append('.')

from jyu.nn.yolo import Conv1d, C2f1d
from jyu.nn.tasks import parse_model, yaml_model_load, guess_model_name, attempt_load_weights

from jyu.utils import LOGGER
from jyu.torch_utils.torch_utils import InitWeight, fuse_conv_and_bn, de_parallel


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
        self.initweightName = initweightName

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
            except(Exception) as e:
                print("except!!!  pass...", e)
                pass
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

    def _split(self):
        for m in self.modules():
            if isinstance(m, C2f1d):
                m.forward = m.forward_split

    def save(self, f):
        '''
        保存权重 params.pt
        f 保存路径, 如: ~/best_params.pt
        '''
        ckpt = {
            "model": deepcopy(de_parallel(self)).half(),
            "args": self.args,
        }
        torch.save(ckpt, f)

    def save_state_dict(self, f):
        '''
        仅保存网络参数
        f 保存路径, 如: ~/best_params.pt
        '''
        state_dict = self.state_dict()
        torch.save(state_dict, f)

    def __iter__(self):
        '''迭代'''
        return iter(self._modules.values())

    def forward(self, input):
        for module in self:
            input = module(input)
        return input

    def forward_bak(self, input):
        '''继承自nn.Module，需要 自定义forward函数'''
        for name, module in self._modules.items():
            input = module(input)
        return input

    def print_model_info(self):
        print("\nprint_model_info: model layers")
        i = 1
        # for module in self.modules():
        for name, module in self._modules.items():
            if type(module) is nn.Sequential:
                for m in module:
                    print(f"  {i:3}  ", m.__class__)
                    i += 1
            else:
                print(f"  {i:3}  ", module.__class__)
                i += 1


from jyu.nn.modules import LSTMBranch
# class Model(nn.Module):
class Model2(nn.Sequential):
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
        self.initweightName = initweightName

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

                self.lstm = model.lstm
                self.fc = model.fc
            except:
                print("except!!!  pass...")
                pass
        else:
            yaml_ = yaml_model_load(yaml_path)
            self.netName = guess_model_name(yaml_)
            ch = yaml_["ch"] = yaml_.get("ch", ch)  # input channels
            if scale:
                yaml_['scale'] = scale
            self.scale, layers, save = parse_model(yaml_, ch, verbose=verbose)
            self.add_layers(layers)

            self.lstm = LSTMBranch(input_size=4096, hidden_size=128, output_features=128)
            # self.fc = nn.Linear(1280, 7)
            self.fc = nn.Linear(1024, 7)
            # self.append(nn.Linear(1280, 7))

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

    def _split(self):
        for m in self.modules():
            if isinstance(m, C2f1d):
                m.forward = m.forward_split

    def save(self, f):
        '''
        保存权重 params.pt
        f 保存路径, 如: ~/best_params.pt
        '''
        ckpt = {
            "model": deepcopy(de_parallel(self)).half(),
            "args": self.args,
        }
        torch.save(ckpt, f)

    def save_state_dict(self, f):
        '''
        仅保存网络参数
        f 保存路径, 如: ~/best_params.pt
        '''
        state_dict = self.state_dict()
        torch.save(state_dict, f)

    def forward(self, input):
        w = input.size(0)
        lstm_output = self.lstm(input)

        for module in self:
            # if module.__class__.__name__ == 'Classify':
            #     input = input + lstm_output
            if type(module) in [LSTMBranch, nn.Linear]:
                continue
            # print(module.__class__.__name__)
            input = module(input)

        input = torch.cat((input, lstm_output), dim=1)  # 合并两个分支的输出

        # tmp = torch.zeros((w, 1280))
        # tmp = torch.zeros((w, input.size(1)))
        # # tmp[:,0:128] = lstm_output
        # tmp[:,-129:-1] = lstm_output
        # lstm_output = tmp
        # lstm_output = lstm_output.to(self.device)
        # input = input + lstm_output
        input = self.fc(input)

        # return input
        return input if self.training else input.softmax(1)



class Model3(nn.Module):
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
        self.initweightName = initweightName
        self.args = {"modelYaml": yaml_path, "fuse": fuse, "split": split}

        self.model1 = Model(yaml_path, weights,
            scale, ch,
            verbose, fuse, split,
            initweightName, device)
        self.scale = self.model1.scale

        self.model1 = Model4()

        self.netName = self.model1.netName

        self.model2 = LSTMBranch(input_size=4096, hidden_size=128, output_features=64)
        self.fc = nn.Linear(1280, 7)

        self.to(device)  # device: cpu, gpu(cuda)

    def save(self, f):
        '''
        保存权重 params.pt
        f 保存路径, 如: ~/best_params.pt
        '''
        ckpt = {
            "model": deepcopy(de_parallel(self)).half(),
            "args": self.args,
        }
        torch.save(ckpt, f)

    def save_state_dict(self, f):
        '''
        仅保存网络参数
        f 保存路径, 如: ~/best_params.pt
        '''
        state_dict = self.state_dict()
        torch.save(state_dict, f)

    def forward(self, input):
        out1 = self.model1(input)
        # out2 = self.model2(input)
        out = self.fc(out1)
        # out = out1
        # return out
        return out if self.training else out.softmax(1)

    def forward2(self, input):
        w = input.size(0)
        # lstm_output = self.lstm(input)

        for module in self:
            # if module.__class__.__name__ == 'Classify':
            #     input = input + lstm_output
            if type(module) in [LSTMBranch, nn.Linear]:
                continue
            # print(module.__class__.__name__)
            input = module(input)

        # input = torch.cat((input, lstm_output), dim=1)  # 合并两个分支的输出

        # tmp = torch.zeros((w, 1280))
        # tmp[:,0:128] = lstm_output
        # tmp[:,-129:-1] = lstm_output
        # lstm_output = tmp
        # lstm_output = lstm_output.to(self.device)
        # input = input + lstm_output
        # input = self.fc(input)

        return input

from jyu.nn.yolo import *
class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.netName = 'Model4'

        self.conv1 = Conv1d(1, 32, 3, 2)
        self.conv2 = Conv1d(32, 64, 3, 2)
        self.c2f1 = C2f1d(64, 64, 1, True)
        self.conv3 = Conv1d(64, 128, 3, 2)
        self.c2f2 = C2f1d(128, 128, 2, True)
        self.scdown1 = SCDown1d(128, 512, 3, 2)
        self.c2f3 = C2f1d(512, 512, 2, True)
        self.scdown2 = SCDown1d(512, 512, 3, 2)
        self.c2fcib = C2fCIB1d(512, 512, True, True)
        self.head = Classify(512, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c2f1(x)
        x = self.conv3(x)
        x = self.c2f2(x)
        x = self.scdown1(x)
        x = self.c2f3(x)
        x = self.scdown2(x)
        x = self.c2fcib(x)
        x = self.head(x)
        return x
