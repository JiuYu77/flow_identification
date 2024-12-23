# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

import sys
sys.path.append('.')
import jyu.utils.transform.transform as tf
from jyu.utils.color_print import print_color

class FlowDataset:
# class FlowDataset(Dataset):
    def __init__(self, datasetPath, sampleLength:int, step:int, transformName=None,
                 clss=None, supervised=True):
        # super().__init__()
        self.datasetPath = datasetPath
        self.sampleLength = sampleLength
        self.step = step
        # self.ptr = 0
        # self.allFile = []
        self.allSample = [] # 所有样本
        self.allLabel = [] # 每个样本 对应的标签； 下标相同
        self.totalSampleNum = 0 # 样本总数
        self.supervised = supervised

        if transformName is not None:
            self.transform = tf.get_transform(transformName)
        else:
            self.transform = None
        print_color(["loading dataset..."])
        if clss is None:
            self._load_data()
        elif clss == -1:
            self._load_data_noLabel()
        else:  # cls 既是文件夹名字，也是类别（标签）
            self._load_data_oneClass(clss)
        self.__len__() # 样本总数

    def sample_to_float(self, data):
        '''将(str字符串)样本, 转换为float样本'''
        sample = []
        for item in data: # 将样本中的每个数据点 由 str 转换为 float
            sample.append(float(item))
        return sample

    def _load_data(self):
        '''加载数据集'''
        datasetPath = self.datasetPath
        sampleLength = self.sampleLength
        step = self.step
        count = 0  # 加载了几个文件
        totalFile = 0  # 文件总数
        dataPointsNum = 0  # 数据点总数
        clsNumList = []  # 每个类别的样本数
        dir = os.path.basename(datasetPath)
        clsNumList, totalFile = self.total()

        for cls in os.listdir(datasetPath):
            label = int(cls)
            clsPath = os.path.join(datasetPath, cls)
            for file in os.listdir(clsPath):
                count += 1
                print(f"\r{'':4}\033[32m{dir:7}\033[0mprogress: {count}/{totalFile}", end='\r')
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
                        sample = self.sample_to_float(sample) # 转为float
                        self.add_sample_and_label(sample, label)
                        break
                    if end > num:
                        sample = lines[num-sampleLength:num]
                        sample = self.sample_to_float(sample) # 转为float
                        self.add_sample_and_label(sample, label)
                        break
                    sample = lines[ptr:end]  # sample列表，存储一个样本，数据点的类型是str
                    sample = self.sample_to_float(sample) # 转为float
                    self.add_sample_and_label(sample, label) # [sample, sample, ...]   [label, label, ...]
                    ptr += step
        print()
        self._print_info(dataPointsNum, clsNumList)

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
                    sample = self.sample_to_float(sample)
                    self.add_sample_and_label(sample, label)
                    break
                if end > num:
                    sample = lines[num-sampleLength:num]
                    sample = self.sample_to_float(sample)
                    self.add_sample_and_label(sample, label)
                    break
                sample = lines[ptr:end]  # sample列表，存储一个样本，数据点的类型是str
                sample = self.sample_to_float(sample)
                self.add_sample_and_label(sample, label)
                ptr += step
        print()
        self.__len__()
        print(f"{'':11}data_points_num: {dataPointsNum}")
        print(f"{'':11}sample_num: {self.totalSampleNum}")

    def _load_data_noLabel(self):
        '''
        加载数据集，只有样本，没有标签
        '''
        datasetPath = self.datasetPath
        sampleLength = self.sampleLength
        step = self.step
        count = 0  # 加载了几个文件
        totalFile = 0  # 文件总数
        dataPointsNum = 0  # 数据点总数
        clsNumList = []  # 每个类别的样本数
        dir = os.path.basename(datasetPath)
        clsNumList, totalFile = self.total()

        for cls in os.listdir(datasetPath):
            label = int(cls)
            clsPath = os.path.join(datasetPath, cls)
            for file in os.listdir(clsPath):
                count += 1
                print(f"\r{'':4}\033[32m{dir:7}\033[0mprogress: {count}/{totalFile}", end='\r')
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
                        sample = self.sample_to_float(sample)
                        self.add_sample(sample)
                        break
                    if end > num:
                        sample = lines[num-sampleLength:num]
                        sample = self.sample_to_float(sample)
                        self.add_sample(sample)
                        break
                    sample = lines[ptr:end]  # sample列表，存储一个样本，数据点的类型是str
                    sample = self.sample_to_float(sample)
                    self.add_sample(sample) # [sample, sample, ...]
                    ptr += step
        print()
        self._print_info(dataPointsNum, clsNumList)

    def add_sample(self, sample):
        self.allSample.append(sample)

    def add_sample_and_label(self, sample, label):
        self.allSample.append(sample)
        self.allLabel.append(label)

    def total(self):
        '''
        return:
          clsNumList 每个类别的样本数
          total 文件总数
        '''
        clsNumList = []
        total = 0
        datasetPath = self.datasetPath
        for cls in os.listdir(datasetPath):
            label = int(cls)
            clsNumList.append(0)
            clsPath = os.path.join(datasetPath, cls)
            for file in os.listdir(clsPath):
                total += 1
        return clsNumList, total
    
    def _print_info(self, dataPointsNum, clsNumList):
        print(f"{'':11}data_points_num: {dataPointsNum}")
        print(f"{'':11}sample_num: {self.__len__()}")
        for i, v in enumerate(clsNumList):
            print(f"{'':15}class {i}: {v}")  # 每个类别样本数

    def __len__(self):
        """返回样本总数"""
        self.totalSampleNum = len(self.allSample)
        return self.totalSampleNum

    def __getitem__(self, index):
        sample = []
        label = None

        data = self.allSample[index]
        sample = np.array([data])
        if self.transform is not None:
            sample = self.transform(sample)
        # else:
        #     totensor = tf.transforms.ToTensor()
        #     sample = totensor(sample)

        label = self.allLabel[index]
        # return sample, label
        if self.supervised:  # 监督
            return sample, label
        else:  # 无监督
            return sample, label, index

    def do_transform(self):
        for i,v in enumerate(self.allSample):
            v = np.array([v])
            x = self.transform(v)
            self.allSample[i] = x

def data_loader(Dataset, datasetPath, sampleLength:int, step:int, transform, clss=None, supervised=True,
                batchSize:int=64, shuffle=True, numWorkers=0):
    dataset = Dataset(datasetPath, sampleLength, step, transform, clss, supervised)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers)
    return dataloader

if __name__ == '__main__':
    datasetPath = "E:\\B_SoftwareInstall\\my_flow\\dataset\\v4\\Pressure\\v4_Pressure_Simple\\4\\train"
    datasetPath = "/home/uu/my_flow/dataset/v4/Pressure/v4_Pressure_Simple/4/val"
    sampleLength = 4096
    step = 2048
    transformName = "normalization_MinMax"
    transformName = "ewt_zScore"
    batchSize = 64
    dataloader = data_loader(FlowDataset, datasetPath, sampleLength, step, transformName, batchSize,shuffle=False)
    # dataloader = data_loader(FlowDataset, datasetPath, sampleLength, step, transformName, batchSize,shuffle=True)
    for i, (samples, labels) in enumerate(dataloader):
        print(samples.shape)
        # print(len(samples[0]))
        # print(samples[0], f"\t\033[31m{i}\033[0m")
        # print(samples[len(samples)-1],f"   {len(samples)}   ", f"\t\033[31m{i}\033[0m")
        # print(samples[1], f"\t\033[31m{i}\033[0m")
        # print(samples[0].dtype)
        print(samples[0].shape)
        # print(type(samples))
        # print(len(samples))
        print(f"\033[32m{labels}\t\033[31m{i}\033[0m")
        # print(len(labels))
        # print(type(labels))
        # print(labels.shape)
        break
        pass
