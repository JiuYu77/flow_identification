# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
import os
from utils import path

def wms_v1_format(sourcePath, outPath):
    """
    wms持液率
    数据集格式化
    去除不必要的数据，如时间
    """
    for dir in os.listdir(sourcePath):
        if dir.startswith('.'):
            continue
        out = os.path.join(outPath, dir)
        path.checkAndInitPath(out)
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

            outFile = os.path.join(out, file)
            with open(outFile, 'w') as fout:
                for item in data:
                    fout.write(str(item) + '\n')


if __name__ == '__main__':
    sourceRoot = '/home/uu/v1'
    outRoot = '../dataset/v1/wms'

    sourcePath = os.path.join(sourceRoot, 'train')
    outPath = os.path.join(outRoot, 'train')
    wms_v1_format(sourcePath, outPath)
    sourcePath = os.path.join(sourceRoot, 'val')
    outPath = os.path.join(outRoot, 'val')
    wms_v1_format(sourcePath, outPath)