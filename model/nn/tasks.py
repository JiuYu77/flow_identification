# -*- coding:UTF-8 -*-
import contextlib
from utils import LOGGER
from torch import nn
from pathlib import Path
import sys
sys.path.append('.')

from utils import ph, LOGGER, colorstr, cfg
from utils.ops import make_divisible

from model.nn.modules import Conv1d, C2f1d, Classify, C2fCIB1d, SCDown1d, PSA1d
from model.nn.other_modules import SENet1d


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
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
                c2 = make_divisible(min(c2, max_channels) * width, 8)

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

    # return scale, nn.Sequential(*layers), sorted(save)
    return scale, layers, sorted(save)

def yaml_model_load(path:str):
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
        d["scale"] = guess_model_scale(path)
        d["yaml_file"] = str(path)
        return d

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

def guess_model_name(model):
    """
    Guess the name of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Name of the model ('YOLOv8_1D', 'YOLOv10_1D').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2name(cfg):
        """Guess from YAML dictionary."""
        m = cfg["net_name"]  # output module name
        return m

    # Guess from model cfg
    if isinstance(model, dict):
        try:
            return cfg2name(model)
        except Exception:
            pass

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            try:
                return eval(x)["task"]
            except Exception:
                pass
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            try:
                return cfg2name(eval(x))
            except Exception:
                pass

        for m in model.modules():
            # if isinstance(m, YOLOv8_1D):
            #     return "YOLOv8_1D"
            # elif isinstance(m, YOLOv10_1D):
            #     return "YOLOv10_1D"
            n = m.__class__.__name__
            if n is"YOLOv8_1D":
                return "YOLOv8_1D"
            elif n == "YOLOv10_1D":
                return "YOLOv10_1D"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "yolov8" in model.stem:
            return "YOLOv8_1D"
        elif "yolov10" in model.stem:
            return "YOLOv10_1D"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model name, assuming 'name=YOLO1d'. "
        "Explicitly define name for your model, i.e. 'task=YOLO1D', 'YOLOv8_1D', 'YOLOv10_1D'."
    )
    return "YOLO1d"  # assume YOLO1d
