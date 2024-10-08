# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

import sys
sys.path.append('.')
import utils.transform.transform as tf
from utils.color_print import print_color

class FlowDataset:
# class FlowDataset(Dataset):
    def __init__(self, datasetPath, sampleLength:int, step:int, transformName, cls:str=None) -> None:
        # super().__init__()
        self.datasetPath = datasetPath
        self.sampleLength = sampleLength
        self.step = step
        # self.ptr = 0
        # self.allFile = []
        self.allSample = []
        self.transform = tf.get_transform(transformName)
        self.totalSampleNum = 0
        print_color(["loading dataset..."])
        if cls is None:
            self._load_data()
        else:  # cls 既是文件夹名字，也是类别（标签）
            self._load_data_oneClass(cls)

    def _load_data(self):
        '''加载数据集'''
        datasetPath = self.datasetPath
        sampleLength = self.sampleLength
        step = self.step
        count = 0  # 加载了几个文件
        total = 0  # 文件总数
        dataPointsNum = 0  # 数据点总数
        clsNumList = []  # 每个类别的样本数
        dir = os.path.basename(datasetPath)
        for cls in os.listdir(datasetPath):
            label = int(cls)
            clsNumList.append(0)
            clsPath = os.path.join(datasetPath, cls)
            for file in os.listdir(clsPath):
                total += 1

        for cls in os.listdir(datasetPath):
            label = int(cls)
            clsPath = os.path.join(datasetPath, cls)
            for file in os.listdir(clsPath):
                count += 1
                print(f"\r{'':4}\033[32m{dir:7}\033[0mprogress: {count}/{total}", end='\r')
                filePath = os.path.join(clsPath, file)
                with open(filePath, 'r') as f:
                    lines = f.readlines()
                ptr = 0
                num = len(lines)
                dataPointsNum += num
                while True:
                    end = ptr + sampleLength
                    clsNumList[label] += 1
                    if num < sampleLength:
                        print(f"\033[31m The number of file data points less than sampleLength\033[0m {filePath}  {num}")
                        sample = lines[:]
                        while len(sample) < sampleLength:
                            sample.append(0)  # 以 0 填充缺少的数据点
                        self.allSample.append((sample, label))
                        break
                    if end > num:
                        sample = lines[num-sampleLength:num]
                        self.allSample.append((sample, label))
                        # self.allSample.append({"sample":sample, "label":label})
                        break
                    sample = lines[ptr:end]  # sample列表，存储一个样本，数据点的类型是str
                    self.allSample.append((sample, label))  # [(sample, label), (sample, label), ...]; sample: [][0], label: [][1]
                    # self.allSample.append({"sample":sample, "label":label})
                    ptr += step
        print()
        self.totalSampleNum = len(self.allSample)
        print(f"{'':11}data_points_num: {dataPointsNum}")
        print(f"{'':11}sample_num: {self.totalSampleNum}")
        for i, v in enumerate(clsNumList):
            print(f"{'':15}class {i}: {v}")  # 每个类别样本数

    def _load_data_oneClass(self, cls):
        '''
        加载数据集的某一类
        Args:
          cls 既是文件夹名字，也是类别
        '''
        datasetPath = self.datasetPath
        sampleLength = self.sampleLength
        step = self.step
        count = 0  # 加载了几个文件
        total = 0  # 文件总数
        dataPointsNum = 0  # 数据点总数
        dir = os.path.basename(datasetPath)
        label = int(cls)
        clsPath = os.path.join(datasetPath, cls)
        for file in os.listdir(clsPath):
            total += 1

        for file in os.listdir(clsPath):
            count += 1
            print(f"\r{'':4}\033[32m{dir:7}\033[0mprogress: {count}/{total}", end='\r')
            filePath = os.path.join(clsPath, file)
            with open(filePath, 'r') as f:
                lines = f.readlines()
            ptr = 0
            num = len(lines)
            dataPointsNum += num
            while True:
                end = ptr + sampleLength
                if num < sampleLength:
                    print(f"\033[31m The number of file data points less than sampleLength\033[0m {filePath}  {num}")
                    sample = lines[:]
                    while len(sample) < sampleLength:
                        sample.append(0)  # 以 0 填充缺少的数据点
                    self.allSample.append((sample, label))
                    break
                if end > num:
                    sample = lines[num-sampleLength:num]
                    self.allSample.append((sample, label))
                    # self.allSample.append({"sample":sample, "label":label})
                    break
                sample = lines[ptr:end]  # sample列表，存储一个样本，数据点的类型是str
                self.allSample.append((sample, label))  # [(sample, label), (sample, label), ...]; sample: [][0], label: [][1]
                # self.allSample.append({"sample":sample, "label":label})
                ptr += step
        print()
        self.totalSampleNum = len(self.allSample)
        print(f"{'':11}data_points_num: {dataPointsNum}")
        print(f"{'':11}sample_num: {self.totalSampleNum}")

    def __len__(self):
        """返回样本总数"""
        return self.totalSampleNum

    def __getitem__(self, index):
        sample = []
        label = None

        # data = self.allSample[index]['sample']
        data = self.allSample[index][0]
        for item in data: # 将样本中的每个数据点 由 str 转换为 float
            sample.append(float(item))
        sample = np.array([sample])
        sample = self.transform(sample)
        # label = self.allSample[index]['label']
        label = self.allSample[index][1]
        return sample, label

def data_loader(Dataset, datasetPath, sampleLength:int, step:int, transform, batchSize:int, shuffle=True, numWorkers=0, cls=None):
    dataset = Dataset(datasetPath, sampleLength, step, transform, cls)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers)
    return dataloader

if __name__ == '__main__':
    datasetPath = "E:\\B_SoftwareInstall\\my_flow\\dataset\\v4\\Pressure\\v4_Pressure_Simple\\4\\train"
    datasetPath = "/home/uu/my_flow/dataset/v4/Pressure/v4_Pressure_Simple/4/train"
    sampleLength = 4096
    step = 2048
    transformName = "normalization_MinMax"
    transformName = "fft_std"
    batchSize = 64
    dataloader = data_loader(FlowDataset, datasetPath, sampleLength, step, transformName, batchSize,shuffle=False)
    # dataloader = data_loader(FlowDataset, datasetPath, sampleLength, step, transformName, batchSize,shuffle=True)
    for i, (samples, labels) in enumerate(dataloader):
        # print(len(samples[0]))
        # print(samples[0], f"\t\033[31m{i}\033[0m")
        # print(samples[len(samples)-1],f"   {len(samples)}   ", f"\t\033[31m{i}\033[0m")
        # print(samples[1], f"\t\033[31m{i}\033[0m")
        # print(samples[0].dtype)
        # print(samples[0].shape)
        # print(type(samples))
        # print(samples.shape)
        # print(len(samples))
        print(f"\033[32m{labels}\t\033[31m{i}\033[0m")
        # print(len(labels))
        # print(type(labels))
        # print(labels.shape)
        break
        pass
