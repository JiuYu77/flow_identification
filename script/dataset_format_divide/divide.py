# -*- coding: utf-8 -*-
import os

def checkAndInitPath(path):
    if not type(path) == list:
        path = [path]
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"\033[33m创建文件夹:\033[0m{p}")


def dataset_divide(formatData, outPath, cls, fileName, test=True):
    """
    划分数据集
    训练集
    验证集
    测试集
    """
    trainSetRadio = 0.7
 
    trainPath = os.path.join(outPath, 'train')  # 训练集输出目录
    valPath = os.path.join(outPath, 'val')  # 验证集输出目录
    checkAndInitPath([trainPath, valPath])

    if test:  # 需要测试集
        trainSetRadio = 0.8
        valSetRadio = 0.1
        testPath = os.path.join(outPath, 'test')
        checkAndInitPath(testPath)

        l = len(formatData)  # 数据总数
        trainNum = int(l * trainSetRadio)  # 训练集数据的个数
        trainSet = formatData[:trainNum]
        valNum = int(l*valSetRadio)
        valSet = formatData[trainNum:trainNum + valNum]
        testSet = formatData[trainNum + valNum:]

        testCLS = os.path.join(testPath, cls)
        checkAndInitPath(testCLS)
        testFile = os.path.join(testCLS, fileName)
        with open(testFile, 'w') as tf:
            for i,d in enumerate(testSet):
                ss = str(d)
                if i != len(testSet)-1:
                    ss = ss + '\n'
                tf.write(ss)
    else:
        l = len(formatData)
        trainNum = int(l * trainSetRadio)
        trainSet = formatData[:trainNum]
        valSet = formatData[trainNum:]

    trainCLS = os.path.join(trainPath, cls)
    valCLS = os.path.join(valPath, cls)
    checkAndInitPath([trainCLS, valCLS])
    trainFile = os.path.join(trainCLS, fileName)
    valFile = os.path.join(valCLS, fileName)
    with open(trainFile, 'w') as tf, open(valFile, 'w') as vf:
        for i,d in enumerate(trainSet):
            ss = str(d)
            if i != len(trainSet)-1:
                ss = ss + '\n'
            tf.write(ss)
        for i,d in enumerate(valSet):
            ss = str(d)
            if i != len(valSet)-1:
                ss = ss + '\n'
            vf.write(ss)

