# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import sys
sys.path.append('.')
import os
import yaml

from utils import FlowDataset, tm, ph, plot

def draw(x, y, outPath):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    fontdict=dict(fontsize=14)
    # ax.set_title("222222", fontdict)
    ax.set_xlabel("s", fontdict)
    ax.set_ylabel("kPa", fontdict)

    ax.plot(x,y)
    fig.savefig(outPath)
    plt.close(fig)

def analysis(dataset, idxList, resultPath, transform):
    for idx in idxList:
        y = None
        x = []

        label = dataset.allSample[idx][1]
        print(idx, label)

        sample, _ = dataset.__getitem__(idx)

        if transform is None:
            print(transform)
            sample = sample[0];y = sample  # None
        elif transform in ["dwt", "fft"]:
            print(transform)
            sample = sample[0];y = sample  # dwt  fft
        elif "ewt" in transform:
            print(transform)
            # sample = sample[0];y = sample  # ewt
            sample = sample[0];y = sample[::2]  # ewt, 采样：数据点数减半
        y_length = len(y)
        print("length_after_transform:", y_length)

        i = 0
        timeStep = 0.001 # 1000Hz ==> 0.001s
        for _ in range(0, y_length):
            i += timeStep
            x.append(i)

        # 画图
        name = str(label) + ".png"
        outPath = os.path.join(resultPath, name)
        draw(x, y, outPath)
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
    resultPath = os.path.join(resultPath, tm.get_result_dir())
    ph.checkAndInitPath(resultPath)
    info_fp_path = os.path.join(resultPath, "info.yaml")

    dataset = FlowDataset(dataPath, length, step, transformName=transform)

    # dataset.__getitem__(0)
    # 1000Hz ==> 0.001s 采样频率

    y_length = analysis(dataset, idxList, resultPath, transform)

    info['length_after_transform'] = y_length
    yaml.dump(info, open(info_fp_path, "w"), sort_keys=False)


if __name__ == '__main__':
    valDataPath = "../dataset/v4/Pressure/v4_Pressure_Simple/4/val"
    trainDataPath = "../dataset/v4/Pressure/v4_Pressure_Simple/4/train"

    # 1000Hz ==> 0.001s 采样频率
    valList = [0, 133, 766, 965, 1166, 1502, 1708]  # val 4096 2048
    valList = [0, 141, 786, 987, 1201, 1547, 1763]  # val 2048 2048
    trainList = [0, 5259]  # train

    length, step = 4096, 2048
    length, step = 2048, 2048
    transform = "ewt"  # None "ewt_std" "ewt" "dwt"

    resultPath = os.path.join("result", "data_analysis", tm.get_result_dir())
    do(resultPath, valDataPath, length, step, None, valList, False)
    do(resultPath, valDataPath, length, step, "ewt", valList, False)
    do(resultPath, valDataPath, length, step, "dwt", valList, False)


