# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
from divide import dataset_divide
import time

def diffPressureData_format_and_divide(sourcePath, outPath, test=True):
    """
    数据集格式化，压差
    压力数据，取压差，即不同列（每个文件中有4列压力数据）的压力做差
    去除不必要的数据：第一列的时间；
    缺失数据处理
    """
    for cls in os.listdir(sourcePath):
        print(cls)
        clsPath = os.path.join(sourcePath, cls)
        for file in os.listdir(clsPath):
            filePath = os.path.join(clsPath, file)
            print(f"\033[34m处理文件：\033[0m{filePath}")
            with open(filePath, 'r') as f:
                lines = f.readlines()
                data = []
                for i,line in enumerate(lines):
                    try:
                        items = line.split('\t')
                        # print(items)
                        if len(items) != 5:
                            raise Exception(f"\033[31m字段个数不足，\033[0m应为5，实际为{len(items)}，{file}：行{i}")
                        press1 = float(items[-2])
                        press2 = float(items[-1])
                        data.append(round(press1 - press2, 5))
                    except Exception as e:
                        # print(e)
                        continue
            dataset_divide(data,outPath, cls, file, test)

if __name__ == '__main__':
    sourcePath = "E:\\B_SoftwareInstall\\my_flow\\流型识别数据集\\v4\\Pressure\\v4_Pressure_Source_A"
    outPath = "E:\\B_SoftwareInstall\\my_flow\\dataset\\v4\\Pressure\\v4_DiffPressure_Simple"
    diffPressureData_format_and_divide(sourcePath, outPath)
