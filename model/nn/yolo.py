# -*- coding: UTF-8 -*-
import contextlib
from torch import nn
import torch
import math
from pathlib import Path
import sys
sys.path.append('.')
from utils import ph, LOGGER, colorstr, cfg
from model.nn.modules import Conv1d, C2f1d, Classify, C2fCIB1d, SCDown1d, PSA1d
from model.nn.other import SENet1d
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
        yaml_ = self.yaml_model_load(yaml_path)
        ch = yaml_["ch"] = yaml_.get("ch", ch)  # input channels
        if scale:
            yaml_['scale'] = scale
        self.scale, layers, save = self.parse_model(yaml_, ch, verbose=verbose)
        self.add_layers(layers)

        if weights:  # 在已有权重上继续训练 or 测试 or 分类(推理, 实际使用)
            assert Path(weights).is_file(), f"{weights} does not exist."
            self.model = torch.load(weights, map_location=device)
            self.load_state_dict(self.model)

    @staticmethod
    def make_divisible(x, divisor):
        """Returns nearest x divisible by divisor."""
        if isinstance(divisor, torch.Tensor):
            divisor = int(divisor.max())  # to int
        return math.ceil(x / divisor) * divisor

    def add_layers(self, layers):
        for idx, module in enumerate(layers):
            self.add_module(str(idx), module)

    @staticmethod
    def guess_model_scale(model_path):
        """
        Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
        uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
        n, s, m, l, or x. The function returns the size character of the model scale as a string.

        Args:
            model_path (str | Path): The path to the YOLO model's YAML file.

        Returns:
            (str): The size character of the model's scale, which can be n, s, m, l, or x.
        """
        with contextlib.suppress(AttributeError):
            import re

            return re.search(r"yolov\d_1D+([nsmlx])", Path(model_path).stem).group(1)  # n, s, m, l, or x
        return ""  # 若yaml文件名中没有nsmlx之一，则返回空字符串

    def yaml_model_load(self, path:str):
        """Load a YOLOv8 model from a YAML file."""
        import re

        path = Path(path)
        if path.stem in (f"yolov{d}_1D{x}6-cls" for x in "nsmlx" for d in (5, 8)):
            new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
            LOGGER.warning(f"WARNING ⚠️ JiuYu77 YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
            path = path.with_name(new_stem + path.suffix)

        unified_path = re.sub(r"(\d+)(_1D)([nslmx])(.+)?$", r"\1\2\4", str(path))  # i.e. yolov8_1Dx-cls.yaml -> yolov8_1D-cls.yaml
        yaml_file = ph.check_yaml(unified_path, hard=False) or ph.check_yaml(path)
        d = cfg.yaml_load(yaml_file)  # model dict
        d["scale"] = self.guess_model_scale(path)
        d["yaml_file"] = str(path)
        return d


    def parse_model(self, d, ch, verbose=True):  # model_dict, input_channels(3)
        import ast

        max_channels = float("inf")
        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        if scales:
            scale = d.get("scale")
            if not scale:  # 没有设置scale
                scale = tuple(scales.keys())[0]  # 默认scale, 配置文件中第一个scale：n
                LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv1d.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")  # print

        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
            m = getattr(nn, m[3:]) if "nn." in m else globals()[m]  # get module
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
            n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
            if m in (
                Classify,
                Conv1d,
                C2f1d,
                C2fCIB1d,
                SCDown1d,
                PSA1d,
                SENet1d,
            ):
                c1, c2 = ch[f], args[0]
                if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                    c2 = self.make_divisible(min(c2, max_channels) * width, 8)

                args = [c1, c2, *args[1:]]
                if m in (C2f1d,):
                    args.insert(2, n)  # number of repeats
                    n = 1
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace("__main__.", "")  # module type
            m.np = sum(x.numel() for x in m_.parameters())  # number params
            m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
            if verbose:
                LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)

        # return nn.Sequential(*layers), sorted(save), scale
        return scale, layers, sorted(save)

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


class YOLO1d:
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
