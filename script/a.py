# -*- coding: utf-8 -*-
import sys

import sklearn.metrics
sys.path.append('.')
# from model import *
# from utils.plot import plot
# import matplotlib.pyplot as plt
import numpy as np
# import torch
# import random


# net = YOLOv8_1D('yolov8_1Ds-cls.yaml', fuse_=True)
# print(net.scale, type(net.scale))
# print(net.__class__.__name__)

# print(torch.__version__)

import sklearn
import sklearn.cluster
# from sklearn.externals import joblib
import joblib

from utils import FlowDataset, cfg, tu

def train():
    print('train...')
    print("\033[33mload dataset...\033[0m")
    dataPath, classNum = cfg.get_dataset_info('v4_press4', tu.getDeviceName())
    trainSet = FlowDataset(dataPath[0], 4096, 2048, 'standardization_zScore')
    X_train = []
    y_train = []
    d = []
    for data in trainSet.allSample:
        for item in data[0]:
            d.append(float(item))
        # y_train.append(data[1])
        X_train.append(np.array(d))
        d = []

    testSet = FlowDataset(dataPath[1], 4096, 2048, 'standardization_zScore')
    X_test = []
    y_test = []
    d = []
    for data in testSet.allSample:
        for item in data[0]:
            d.append(float(item))
        # y_test.append(data[1])
        X_test.append(np.array(d))
        d = []

    print("\033[33mKMeans...\033[0m")
    k_means = sklearn.cluster.KMeans(n_clusters=7)
    KMeans = k_means.fit(X_train)
    joblib.dump(k_means, 'a.pkl')

    pre = k_means.predict(X_train)
    # pre = k_means.predict(X_test)
    print(pre)
    score = sklearn.metrics.silhouette_score(X_train, pre)
    print(score)

def test():
    print('test...')
    dataPath, classNum = cfg.get_dataset_info('v4_press4', tu.getDeviceName())
    cls = '1'
    dataSet = FlowDataset(dataPath[0], 4096, 2048, 'standardization_zScore', cls=cls)
    # dataSet = FlowDataset(dataPath[0], 4096, 2048, 'standardization_zScore')
    x = []
    y_train = []
    d = []
    for data in dataSet.allSample:
        for item in data[0]:
            d.append(float(item))
        # y_train.append(data[1])
        x.append(np.array(d))
        d = []

    print("\033[33mKMeans...\033[0m")
    # k_means = sklearn.cluster.KMeans(n_clusters=7)
    k_means = joblib.load('a.pkl')
    pre:np.ndarray = k_means.predict(x)
    print(pre)
    clsList = []
    clsList.append([i for i in range(classNum)])
    clsList.append([0 for i in range(classNum)])
    print(clsList)
    with open('k_means.yaml', 'w') as f:
        f.write('sample_num: ' + str(len(x)) + '\n'*2)
        for v in pre:
            clsList[1][v] += 1
        f.write('predict_info:\n ' + str(np.array(clsList)) + '\n'*2)
        cls = clsList[1].index(max(clsList[1]))
        f.write('accuracy: ' + str(clsList[1][int(cls)]/sum(clsList[1])) + f"  = {clsList[1][int(cls)]} / {sum(clsList[1])}" + '\n')
    score = sklearn.metrics.silhouette_score(x, pre)
    print(score)


# train()
test()
