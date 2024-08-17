# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
from divide import dataset_divide, divide

def pressureData_format_and_divide(sourcePath, outPath, idx, test=True):
    """
    压力数据
    数据集格式化，保留某一列的压力数据
    去除不必要的数据：第1列的时间；
    缺失数据处理
    """
    for cls in os.listdir(sourcePath):
        clsPath = os.path.join(sourcePath, cls)
        for file in os.listdir(clsPath):
            filePath = os.path.join(clsPath, file)
            print(f"\033[34m处理文件：\033[0m{filePath}")
            with open(filePath, 'r') as f:
                lines = f.readlines()
                data = []
                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) != 5:
                            raise Exception(f"\033[31m字段个数不足，\033[0m应为5，实际为{len(items)}，{file}：行")
                        data.append(float(items[idx]))
                        # data.append(round(float(items[idx]), 4))
                    except Exception as e:
                        continue
            dataset_divide(data, outPath, cls, file, test)
 

if __name__ == '__main__':
    sourcePath = "E:\\B_SoftwareInstall\\my_flow\\流型识别数据集\\v4\\Pressure\\v4_Pressure_Source_A"
    outPath = [
            "E:\\B_SoftwareInstall\\my_flow\\dataset\\v4\\Pressure\\v4_Pressure_Simple\\3",
            "E:\\B_SoftwareInstall\\my_flow\\dataset\\v4\\Pressure\\v4_Pressure_Simple\\4",
        ]
    # pressureData_format_and_divide(sourcePath, outPath[0], idx=3, test=False)
    pressureData_format_and_divide(sourcePath, outPath[1], idx=4, test=True)
    # pressureData_format_and_divide(sourcePath, outPath[1], idx=4)
