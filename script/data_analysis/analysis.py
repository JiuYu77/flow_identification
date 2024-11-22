# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import sys
sys.path.append('.')
import os

from utils import FlowDataset, tm, ph

resultPath = os.path.join("result", "data_analysis", tm.get_result_dir())
ph.checkAndInitPath(resultPath)

dataset = FlowDataset("../dataset/v4/Pressure/v4_Pressure_Simple/4/val", 4096, 2048)
# dataset = FlowDataset("../dataset/v4/Pressure/v4_Pressure_Simple/4/train", 4096, 2048)
# dataset = FlowDataset("../dataset/v4/Pressure/v4_Pressure_Simple/4/train", 4096, 2048, transformName="ewt_std")
# dataset = FlowDataset("../dataset/v4/Pressure/v4_Pressure_Simple/4/train", 4096, 2048, transformName="ewt")
# dataset = FlowDataset("../dataset/v4/Pressure/v4_Pressure_Simple/4/train", 4096, 2048, transformName="fft")
# dataset = FlowDataset("../dataset/v4/Pressure/v4_Pressure_Simple/4/train", 4096, 2048, transformName="dwt")

# dataset.__getitem__(0)

# 1000Hz ==> 0.001s
valList = [0, 133, 766, 965, 1166, 1502, 1708]  # val
valList = [0, 133, 766, 965, 1166, 1502, 1708]  # val
trainList = [0, 5259]  # train

for idx in valList:
    label = dataset.allSample[idx][1]
    print(idx, label)
    y = dataset.allSample[idx][0]

    # sample, _ = dataset.__getitem__(700);sample = sample[0];y = sample[::2]  # ewt
    # sample, _ = dataset.__getitem__(650);sample = sample[0];y = sample  # fft  dwt


    x = []

    i = 0
    step = 0.001
    # print(len(y))
    for _ in range(0, len(y)):
        i += step
        x.append(i)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    fontdict=dict(fontsize=14)
    ax.set_title("222222", fontdict)
    ax.set_xlabel("s", fontdict)
    ax.set_ylabel("Pa", fontdict)

    ax.plot(x,y)
    name = str(label) + ".png"
    outPath = os.path.join(resultPath, name)
    fig.savefig(outPath)
