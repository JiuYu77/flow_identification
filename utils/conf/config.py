# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
import re
import yaml


DATASET_CONFIG_PATH = "./conf/dataset"

def yaml_load(file, append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        try:
            data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        except:
            data = yaml.unsafe_load(s) or {}
        if append_filename:
            data["yaml_file"] = str(file)
    return data

class config:
    def __init__(self, path) -> None:
        self.path = path

    def get_yaml(self, datasetName:str):
        """读取数据集配置文件, 如v4_press.yaml"""
        p = Path(self.path)
        files = p.glob(f'{datasetName}.yaml')
        confList = list(files)
        file = confList[0]
        if len(confList) == 1:
            file = confList[0]
            info = yaml_load(file)
        else:
            raise KeyError("Dataset not found")
        return info

def get_dataset_info(datasetName:str, deviceName:str, train=True):
    """
    返回数据集路径 和 类别数量
    datasetName: 数据集的名字, 和文件名相同;
    deviceName: 平台, hy windows linux 或 mac
    """
    config_ = config(DATASET_CONFIG_PATH)
    info = config_.get_yaml(datasetName)
    if deviceName in info.keys():
        datasetPath = info[deviceName]["path"]
        classNum =  info["class_num"]

        trainDatasetPath = Path(datasetPath).joinpath(info[deviceName]['train'])
        valDatasetPath = Path(datasetPath).joinpath(info[deviceName]['val'])
        testDatasetPath = Path(datasetPath).joinpath(info[deviceName]['test'])
    else:
        datasetPath = info["path"]
        classNum =  info["class_num"]

        trainDatasetPath = Path(datasetPath).joinpath(info['train'])
        valDatasetPath = Path(datasetPath).joinpath(info['val'])
        testDatasetPath = Path(datasetPath).joinpath(info['test'])

    if train:
        return [trainDatasetPath, valDatasetPath], classNum, info  # [训练集路径, 验证集路径], 类别数量
    elif train is False:
        return testDatasetPath, classNum  # 测试集路径, 类别数量

def get_classes(filename):
    # classes = ['段塞流', '伪段塞流', '分层波浪流', '分层光滑流', '泡沫段塞流', '分层泡沫波浪流', '泡沫环状流']
    # classesFlag = [0, 1, 2, 3, 4, 5, 6]
    config_ = config(DATASET_CONFIG_PATH)
    info = config_.get_yaml(filename)
    classes, classesFlag = list(info['names'].values()), list(info['names'].keys())
    return classes, classesFlag  # 类别，类别对应的标签

