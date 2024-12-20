# -*- coding: UTF-8 -*-
"""
分析
采样频率 1000Hz ==> 0.001s
"""

import matplotlib.pyplot as plt
import sys
sys.path.append('.')
import os
import yaml

from utils import FlowDataset, tm, ph, plot


def draw(x, y, title, xlabel, ylabel, outPath):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    fontdict=dict(fontsize=14)
    ax.set_title(title, fontdict)
    ax.set_xlabel(xlabel, fontdict)
    ax.set_ylabel(ylabel, fontdict)
    if x is None:
        ax.plot(y)
    else:
        ax.plot(x,y)
    fig.savefig(outPath)
    plt.close(fig)

def analysis(dataset, idxList, resultPath, transform):
    clsList = ["段塞流", "伪段塞流", "分层波浪流", "分层光滑流", "泡沫段塞流", "分层泡沫波浪流", "泡沫环状流"]
    clsList = ["slug flow", "pseudo-slug flow", "stratified wavy flow", "stratified smooth flow",
           "foamy slug flow", "stratified foamy wavy flow", "foamy annular flow"]

    for idx in idxList:
        y = []
        x = []
        xlabel, ylabel = "t/s", "Pressure/kPa"  # 时间/s  压力/kPa

        label = dataset.allSample[idx][1]
        print(idx, label)

        sample, _ = dataset.__getitem__(idx)

        if transform is not None and "ewt" in transform:
            print(transform)
            # sample = sample[0];y = sample  # ewt
            sample = sample[0];y = sample[::2]  # ewt, 采样：数据点数减半
        else:
            print(transform)  # None  dwt  fft
            sample = sample[0];y = sample
        y_length = len(y)
        print("length_after_transform:", y_length)

        i = 0
        Fs = 1000    # 采样频率(数据采集频率) 1000Hz ==> 0.001s
        Ts = 1 / Fs  # 采样周期(数据采集周期)  0.001s
        for _ in range(0, y_length):
            i += Ts
            x.append(i)
        
        if transform is not None and ("ewt" in transform or "e-w-t" in transform
                                      or "dwt" in transform
                                    ):
            # xlabel, ylabel = "", "amplitude A"
            xlabel, ylabel = "", "Pressure/kPa"
            x = None

        # 画图
        name = str(label) + ".png"
        cls = clsList[label]
        outPath = os.path.join(resultPath, name)
        draw(x, y, cls, xlabel, ylabel, outPath)
    return y_length  # 用于绘图的序列的数据点数, 样本长度 或 ewt、dwt等变换后的序列长度


def do(resultPath, dataPath, length, step, transform, idxList, train=True):
    train_or_val = "train" if train else "val"
    info = {
        "dataset": {"path": dataPath, "train_or_val": train_or_val},
        "sampleLength":length, 
        "step": step,
        "transform": transform,
        "length_after_transform": None,
    }
    # resultPath = os.path.join(resultPath, tm.get_result_dir())
    resultPath = os.path.join(resultPath, str(transform))
    ph.checkAndInitPath(resultPath)
    info_fp_path = os.path.join(resultPath, "info.yaml")

    dataset = FlowDataset(dataPath, length, step, transformName=transform)

    # dataset.__getitem__(0)
    # 1000Hz ==> 0.001s 采样频率

    if transform is not None and "ewt" in transform:
        # 降采样
        outPath = os.path.join(resultPath, "downSample")
        ph.checkAndInitPath(outPath)
        info_fp_path = os.path.join(outPath, "info.yaml")

        y_length = analysis(dataset, idxList, outPath, transform)

        info['length_after_transform'] = y_length
        yaml.dump(info, open(info_fp_path, "w"), sort_keys=False)

        # 源
        outPath = os.path.join(resultPath, "origin")
        ph.checkAndInitPath(outPath)
        info_fp_path = os.path.join(outPath, "info.yaml")

        y_length = analysis(dataset, idxList, outPath, "e-w-t")

        info['length_after_transform'] = y_length
        yaml.dump(info, open(info_fp_path, "w"), sort_keys=False)
        return

    y_length = analysis(dataset, idxList, resultPath, transform)

    info['length_after_transform'] = y_length
    yaml.dump(info, open(info_fp_path, "w"), sort_keys=False)



if __name__ == '__main__':
    valDataPath = "../dataset/v4/Pressure/v4_Pressure_Simple/4/val"
    trainDataPath = "../dataset/v4/Pressure/v4_Pressure_Simple/4/train"

    resultPath = os.path.join("result", "data_analysis", tm.get_result_dir())

    length, step = 4096, 2048
    length, step = 2048, 2048
    # transform  None  "ewt"  "dwt"

    trainList = [0, 1095, 6216, 7791, 9479, 12208, 13900]  # train 2048 2048

    valList = [0, 133, 766, 965, 1166, 1502, 1708]  # val 4096 2048
    valList = [0, 144, 786, 987, 1201, 1547, 1763]  # val 2048 2048

    # idxList = trainList
    # flag = True
    # dataPath = trainDataPath

    idxList = valList
    flag = False
    dataPath = valDataPath

    do(resultPath, dataPath, length, step, None,  idxList, flag)
    do(resultPath, dataPath, length, step, "ewt", idxList, flag)
    do(resultPath, dataPath, length, step, "dwt", idxList, flag)

