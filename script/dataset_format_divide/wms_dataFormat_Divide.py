# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
from divide import dataset_divide


def wmsData_format_and_divide(sourcePath, outPath, test):
    """
    wms持液率
    数据集格式化
    去除不必要的数据，如时间
    """
    assert test == 1 or test == 0, f'test ValueError, except 0 or 1, but got {test}'
    test = test == 1 or not test == 0
    for dir in os.listdir(sourcePath):
        if dir.startswith('.'):
            continue
        clsPath = os.path.join(sourcePath, dir)
        for file in os.listdir(clsPath):
            if file.startswith('.'):
                continue
            filePath = os.path.join(clsPath, file)
            print(f"\033[34m处理文件：\033[0m{filePath}")
            with open(filePath) as fin:
                lines = fin.readlines()[2:]
            data = []
            for line in lines:
                data.append(float(line.split(' ')[-1]))
            dataset_divide(data, outPath, dir, file, test)


if __name__ == '__main__':
    sourcePath = 'E:\\B_SoftwareInstall\\my_flow\\流型识别数据集\\v4\\WMS\\v4_WMS_Source_A'
    outPath = 'E:\\B_SoftwareInstall\\my_flow\\流型识别数据集\\v4\\WMS\\v4_WMS_Simple'

    parser = ArgumentParser()
    parser.add_argument('-s', '--sourcePath', type=str, default=sourcePath)
    parser.add_argument('-o', '--outPath', type=str, default=outPath)
    # parser.add_argument('-t', '--test', type=bool, default=False)  # 默认值为False，命令行中 -t 指定的参数都会被解析为True
    # parser.add_argument('-t', '--test', type=bool, default=True)  # 默认值为True，命令行中 -t 指定的参数都会被解析为True
    # parser.add_argument('-t', '--test', action='store_true', default=False)  # 默认值为False，命令行中 -t即为True，只需要输入-t
    parser.add_argument('-t', '--test', type=int, default=1, help='1 is True, 0 is False')
    opt = parser.parse_args()
    print(f"\033[31m{opt.test}\033[0m")

    wmsData_format_and_divide(**vars(opt))
