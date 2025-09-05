# -*- coding: UTF-8 -*-
import sys
sys.path.append('.')
from jyu.rknn.rknnlite import RknnLite, RKNNLite
from jyu.dataset.flowDataset import FlowDataset
from jyu.dataloader.flowDataLoader import FlowDataLoader
import numpy as np
from jyu.utils import tm
from jyu.rknn.utils import accuracy


datasetPath = "/home/orangepi/flow/dataset/flow/v4/Pressure/4/val"
sampleLength = 4096
step = 2048
transformName = "zScore_std"
batchSize = 31
shuffle=True
dataset = FlowDataset(datasetPath, sampleLength, step, transformName, clss=None, supervised=True, toTensor=False)
dataloader = FlowDataLoader(dataset, batchSize, shuffle)
print("sample_num:", len(dataset))
print("batch_num:", len(dataloader))


rknn_model = "result/rknn/20250603.175044_YI-Netv1/YI-Netv1-non_dynamic_axes.rknn"
rknn_model = "result/rknn/20250603.175044_YI-Netv1/YI-Netv1-dynamic_axes.rknn"
rknn_lite = RknnLite(rknn_model, RKNNLite.NPU_CORE_0)

acc_num = 0
total_sample_num = 0
timer = tm.Timer()

for X, Y in dataloader:
    X = X.astype(np.float32)
    timer.start()
    outputs = rknn_lite(X) # 推理
    # 后处理（示例：分类模型）
    pred_label = np.argmax(outputs, axis=2)
    y_hat = np.array(outputs)
    acc_num += accuracy(y_hat, Y)
    timer.stop()
    total_sample_num += len(Y)
    print("Predicted class:", pred_label, "batch_acc_num", accuracy(y_hat, Y), "acc_num:", acc_num)


print("total_sample_num:", total_sample_num)
print("total_acc_num:", acc_num)
print("acc:", acc_num/total_sample_num)
print("time:", timer.sum(), "sec")
print("speed", total_sample_num/timer.sum(), "samples/sec")

rknn_lite.release()
