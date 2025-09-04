# -*- coding: utf-8 -*-
import numpy as np


class FlowDataLoader:
    def __init__(self, dataset, batchSize, shuffle:bool=True):
        self.dataset = dataset
        self.batchSize = batchSize
        self.shuffle = shuffle

        self.sampleNum = len(dataset)
        self.index = 0

        if shuffle:
            self.indices = np.arange(self.sampleNum)
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        X, Y = [], []
        if not self.shuffle:
            for i in range(self.batchSize):
                if self.index >= self.sampleNum:
                    raise StopIteration
                x, y = self.dataset[self.index]
                self.index += 1
                X.append(x)
                Y.append(y)
        else:
            for i in range(self.batchSize):
                if self.index >= self.sampleNum:
                    raise StopIteration
                idx = self.indices[self.index]
                x, y = self.dataset[idx]
                self.index += 1
                X.append(x)
                Y.append(y)

        if isinstance(X[0], np.ndarray):
            X = np.array(X)
        if isinstance(X[0], np.ndarray):
            Y = np.array(Y)
        return X, Y
    
    def __len__(self):
        '''返回每个epoch中的batch数量'''
        return (self.sampleNum + self.batchSize - 1) // self.batchSize


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from jyu.dataset.flowDataset import FlowDataset
    datasetPath = "/root/my_flow/dataset/flow/v4/Pressure/4/val"
    sampleLength = 4096
    step = 2048
    transformName = "zScore_std"
    batchSize = 64
    shuffle=True
    dataset = FlowDataset(datasetPath, sampleLength, step, transformName, clss=None, supervised=True, toTensor=False)
    dataloader = FlowDataLoader(dataset, batchSize, shuffle)

    print("len(dataloader)=", len(dataloader))

    for i, (X, Y) in enumerate(dataloader):
        if i >= 6:
            break
        print(X.shape)
        print(Y.shape)
        print(Y)

