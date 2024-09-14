# -*- coding: UTF-8 -*-
from pathlib import Path
import sys
sys.path.append('.')
from utils import ROOT
import glob
import os

def checkAndInitPath(path):
    """创建文件夹 或 路径"""
    if not type(path) == list:
        path = [path]
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"\033[33m创建文件夹:\033[0m{p}")

def check_suffix(file, suffix, msg=''):  # optional
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"

def check_file(file, suffix='', hard=True):
    """Search file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    # files = glob.glob(str(ROOT / "conf" / "**" / file), recursive=True)  # find file
    files = glob.glob(str(ROOT / "conf" / "models" / "**" / file), recursive=True)  # find file
    if not files and hard:
        raise FileNotFoundError(f"'{file}' does not exist")
    elif len(files) > 1 and hard:
        raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
    return files[0] if len(files) else []  # return file

def check_yaml(file, suffix=(".yaml", ".yml"), dir='', hard=True):
    return check_file(file, suffix, hard=hard)


# print(check_yaml('yolov8_1D-cls.yaml'))
# print(check_yaml('yolov8_1D-cls.yaml', dir='yolov8'))
