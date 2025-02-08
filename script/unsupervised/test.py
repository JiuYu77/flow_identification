# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import sys
sys.path.append('.')
from jyu.model import YI
from jyu.nn import MODEL_YAML_DEFAULT
from jyu.utils import FlowDataset, data_loader, tu, cfg, colorstr, print_color, ROOT, ph, tm, plot
import os
import torch
import numpy as np
import yaml


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['v4_press4_unsupervised', 'v4_wms', 'v1_wms'], default='v4_press4_unsupervised', help="数据集的名字，和yaml配置文件名相同，不包括.yaml后缀")
    parser.add_argument('-b', '--batchSize', type=int, default=64, help='Number of batch size to test.')
    parser.add_argument('-sl', '--sampleLength', type=int, default=4096, help='Data Length')
    parser.add_argument('-s', '--step', type=int, default=2048, help='Step Length')
    parser.add_argument('-t', '--transform', type=str,
                        choices=['standardization_zScore',
                                 'normalization_MinMax',
                                 'multiple_zScore',
                                 'multiple_MinMax',
                                 'ewt_std',
                                 'dwt_std',
                                 'dwtg_std',
                                 'fft_std',
                                ],
                        default=None, help='Transform for test sample')
    parser.add_argument('-sf', '--shuffleFlag', type=int, default=1, help='1 is True, 0 is False')
    # parser.add_argument('-n', '--numWorkers', type=int, default=0)
    parser.add_argument('-n', '--numWorkers', type=int, default=4)
    parser.add_argument('-my', '--modelYaml', type=str, default=MODEL_YAML_DEFAULT)



    netPath = os.path.join('result', 'train_unsupervised', '20240825.174548_Yolov8_1D_unsupervised', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20240825.192300_Yolov8_1D_unsupervised', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20240825.203748_Yolov8_1D_unsupervised', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20240825.211405_Yolov8_1D_unsupervised', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20240825.213245_Yolov8_1D', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20240825.215013_Yolov8_1D', 'weights', '50_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20240825.225646_Yolov8_1D', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20240827.153703_Yolov8_1D', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20241223.134745_YOLOv8_1D', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20241223.143508_YOLOv8_1D', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20241223.155108_YOLOv8_1D', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20250102.214245_YOLOv8_1D', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20250103.095008_YOLOv8_1D', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20250111.154535_YOLOv8_1D', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train_unsupervised', '20250111.165554_YOLOv8_1D', 'weights', 'best_params.pt')

    parser.add_argument('-w', '--weights', type=str, default=netPath)
    # parser.add_argument('--cls', type=str, default=None)

    opt = parser.parse_args()
    return opt

def test_one(dataset,
         batchSize,
         sampleLength,
         step,
         transform:str,
         shuffleFlag,
         numWorkers,
         modelYaml,
         weights,
         cls,
         thisDir
         ):
    assert shuffleFlag == 1 or shuffleFlag == 0, f'shuffle_flag ValueError, except 0 or 1, but got {shuffleFlag}'
    shuffle = shuffleFlag == 1 or not shuffleFlag == 0
    device = tu.get_device()

    # 读取训练信息
    p = os.path.dirname(weights)
    path = os.path.dirname(p)
    yml = cfg.yaml_load(os.path.join(path, 'info.yaml'))
    # transform
    transform = yml['transform'] if transform is None else transform
    if transform.lower().find('noise') != -1:
        transform = 'standardization_zScore'
 
    # 数据集
    deviceName = tu.getDeviceName()
    if deviceName == "windows":
        numWorkers = 0
    testDatasetPath, classNum = cfg.get_dataset_info(dataset, deviceName, train=False)
    testIter = data_loader(FlowDataset, testDatasetPath, sampleLength, step, transform,  clss=cls,
                           batchSize=batchSize, shuffle=shuffle, numWorkers=numWorkers)

    # 模型
    print('loading model...')
    scale = yml['model_settings']['model_scale']
    # modelYaml = modelYaml if modelYaml else yml['model_settings']['modelYaml']
    fuse_, split_ = yml['model_settings']['fuse'], yml['model_settings']['split']
    net = YI(modelYaml, weights, scale=scale, fuse=fuse_, split=split_, device=device)
    net.eval()
    netName = net.__class__.__name__
    modelParamAmount = sum([p.nelement() for p in net.parameters()])

    # 路径
    resultPath = os.path.join(ROOT, 'result', 'test_unsupervised', thisDir, 'new_labels')
    ph.checkAndInitPath(resultPath)
    # cmPath = os.path.join(resultPath, 'confusionMatrix.txt')
    cmPath = os.path.join(resultPath, f'confusionMatrix_{cls}.yaml')
    confusionMatrixPath = os.path.join(resultPath, f'confusionMatrix_{cls}.png')
    info_fp_path = os.path.join(resultPath, 'info.yaml')

    task_info = {
        "dataset": dataset,
        "task_name": "Model training",
        "model_name": netName,
        "model_scale": net.scale,
        "batch_size": batchSize,
        "sample_length": sampleLength,
        "step": step,
        "model_parameter_amount": f"{modelParamAmount / 1e6:.3f}M",
        "transform": transform,
        "shuffle": shuffle,
        "numWorkers": numWorkers,
        "net_path": weights,
        "test_time_consuming": None,
        "speed": None,
        "learn_method": "Unsupervised Learning",
    }
    yaml.dump(task_info, open(info_fp_path, "w"))

    totalCorrect = 0
    totalNum = 0
    classes, classesFlag = cfg.get_classes(dataset)  # 类别list, 类别对应的整数标签list
    fontsize = 12 if len(classesFlag) == 7 else 15  # 7种类别 12, 4种类别 15
    confusionMatrix = plot.ConfusionMatrix(classesFlag, normalize=True, figsize=(6, 5), fontdict=dict(fontsize=fontsize))  # fontsize=12 15
    sampleNum = [[i for i in range(classNum)],[0 for _ in range(classNum)]]  # 每种流型的样本数
    print("-----------------------------------------")
    print(f"|{colorstr('yellow', ' Start testing:')}")
    print(f"|{colorstr('green', ' testing device:')} {device}")
    print("-----------------------------------------")
    testTimer = tu.Timer()
    print(f"{colorstr('blue', 'timer start...')}")
    testTimer.start()

    print_color(["bright_green", "preparing data..."])
    for i, (X,y) in enumerate(testIter):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        correctNum = tu.accuracy(y_hat, y)  # 预测正确的数量
        totalCorrect += correctNum
        totalNum += len(y)
        y_hat_ = torch.argmax(y_hat, dim=1)
        # print(y_hat_.shape)
        # print(y_hat_ == y, correctNum)
        print(f"\rbatch {i}", end='\r')
        for i in range(len(y_hat_)):
            trueLabel, preLabel = int(y[i]), int(y_hat_[i])
            confusionMatrix.add(trueLabel, preLabel)
            sampleNum[1][trueLabel] += 1

    testTimer.stop()
    print(f"\n{colorstr('blue', 'timer stop...')}")
    tSec = testTimer.sum()  # 秒数

    totalCorrect = int(totalCorrect)
    acc = round(totalCorrect / totalNum, 8)

    timeConsuming = tm.sec_to_HMS(tSec)
    task_info["test_time_consuming"] = timeConsuming
    speed = int(totalNum / tSec)
    task_info["speed"] = f"{speed} samples/sec"
    yaml.dump(task_info, open(info_fp_path, "w"))

    print('total_correct:', totalCorrect, '   total_num:', totalNum, '   acc:', acc)

    print("-----------------------------------------")
    print(f"|{colorstr('yellow', ' End testing:')}")
    print(f"| test time consuming: {timeConsuming}")  # 训练耗时
    print(f"| speed: {speed} samples/sec")  # speed
    print("-----------------------------------------")

    # 混淆矩阵
    confusionMatrix.draw_save(confusionMatrixPath, dpi=80)
    # 保存测试结果
    with open(cmPath, 'w', encoding='utf-8') as f:
        f.write('Accuracy: ' + str(acc) + "\n"*2)
        f.write('Correct num: ' + str(totalCorrect) + "\n"*2)
        f.write('total sample num: ' + str(totalNum) + '\n'*2)
        f.write('sample num:\n' + str(np.array(sampleNum)) + '\n'*2)
        f.write('confusionMatrix:\n' + str(confusionMatrix.cm))

def get_new_label(opt):
    classNum = 7
    for i in range(classNum):
        opt.cls = str(i)
        test_one(**vars(opt))

def test(dataset,
         batchSize,
         sampleLength,
         step,
         transform:str,
         shuffleFlag,
         numWorkers,
         modelYaml,
         weights,
         thisDir
         ):
    assert shuffleFlag == 1 or shuffleFlag == 0, f'shuffle_flag ValueError, except 0 or 1, but got {shuffleFlag}'
    shuffle = shuffleFlag == 1 or not shuffleFlag == 0
    device = tu.get_device()

    # 读取训练信息
    path = os.path.dirname(weights)
    yml = cfg.yaml_load(os.path.join(path, 'info.yaml'))
    # transform
    transform = yml['transform'] if transform is None else transform
    if transform.lower().find('noise') != -1:
        transform = 'standardization_zScore'
 
    # 数据集
    deviceName = tu.getDeviceName()
    if deviceName == "windows":
        numWorkers = 0
    testDatasetPath, classNum = cfg.get_dataset_info(dataset, deviceName, train=False)
    testIter = data_loader(FlowDataset, testDatasetPath, sampleLength, step, transform,
                           batchSize, shuffle=shuffle, numWorkers=numWorkers)

    # 模型
    print('loading model...')
    scale = yml['model_settings']['model_scale']
    # modelYaml = modelYaml if modelYaml else yml['model_settings']['modelYaml']
    fuse_, split_ = yml['model_settings']['fuse_'], yml['model_settings']['split_']
    net = YI(modelYaml, weights, scale=scale, fuse_=fuse_, split_=split_, device=device)
    net.eval()
    netName = net.__class__.__name__
    modelParamAmount = sum([p.nelement() for p in net.parameters()])

    # 路径
    resultPath = os.path.join(ROOT, 'result', 'test_unsupervised', thisDir, netName)
    ph.checkAndInitPath(resultPath)
    # cmPath = os.path.join(resultPath, 'confusionMatrix.txt')
    cmPath = os.path.join(resultPath, 'confusionMatrix.yaml')
    confusionMatrixPath = os.path.join(resultPath, 'confusionMatrix.png')
    info_fp_path = os.path.join(resultPath, 'info.yaml')

    task_info = {
        "dataset": dataset,
        "task_name": "Model training",
        "model_name": netName,
        "model_scale": net.scale,
        "batch_size": batchSize,
        "sample_length": sampleLength,
        "step": step,
        "model_parameter_amount": f"{modelParamAmount / 1e6:.3f}M",
        "transform": transform,
        "shuffle": shuffle,
        "numWorkers": numWorkers,
        "net_path": weights,
        "test_time_consuming": None,
        "speed": None,
        "learn_method": "Unsupervised Learning",
    }
    yaml.dump(task_info, open(info_fp_path, "w"))

    totalCorrect = 0
    totalNum = 0
    classes, classesFlag = cfg.get_classes(dataset)  # 类别list, 类别对应的整数标签list
    fontsize = 12 if len(classesFlag) == 7 else 15  # 7种类别 12, 4种类别 15
    confusionMatrix = plot.ConfusionMatrix(classesFlag, normalize=True, figsize=(6, 5), fontdict=dict(fontsize=fontsize))  # fontsize=12 15
    sampleNum = [[i for i in range(classNum)],[0 for _ in range(classNum)]]  # 每种流型的样本数
    print("-----------------------------------------")
    print(f"|{colorstr('yellow', ' Start testing:')}")
    print(f"|{colorstr('green', ' testing device:')} {device}")
    print("-----------------------------------------")
    testTimer = tu.Timer()
    print(f"{colorstr('blue', 'timer start...')}")
    testTimer.start()

    print_color(["bright_green", "preparing data..."])
    for i, (X,y) in enumerate(testIter):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        correctNum = tu.accuracy(y_hat, y)  # 预测正确的数量
        totalCorrect += correctNum
        totalNum += len(y)
        y_hat_ = torch.argmax(y_hat, dim=1)
        # print(y_hat_.shape)
        # print(y_hat_ == y, correctNum)
        print(f"\rbatch {i}", end='\r')
        for i in range(len(y_hat_)):
            trueLabel, preLabel = int(y[i]), int(y_hat_[i])
            confusionMatrix.add(trueLabel, preLabel)
            sampleNum[1][trueLabel] += 1

    testTimer.stop()
    print(f"\n{colorstr('blue', 'timer stop...')}")
    tSec = testTimer.sum()  # 秒数

    totalCorrect = int(totalCorrect)
    acc = round(totalCorrect / totalNum, 8)

    timeConsuming = tm.sec_to_HMS(tSec)
    task_info["test_time_consuming"] = timeConsuming
    speed = int(totalNum / tSec)
    task_info["speed"] = f"{speed} samples/sec"
    yaml.dump(task_info, open(info_fp_path, "w"))

    print('total_correct:', totalCorrect, '   total_num:', totalNum, '   acc:', acc)

    print("-----------------------------------------")
    print(f"|{colorstr('yellow', ' End testing:')}")
    print(f"| test time consuming: {timeConsuming}")  # 训练耗时
    print(f"| speed: {speed} samples/sec")  # speed
    print("-----------------------------------------")

    # 混淆矩阵
    confusionMatrix.draw_save(confusionMatrixPath, dpi=80)
    # 保存测试结果
    with open(cmPath, 'w', encoding='utf-8') as f:
        f.write('Accuracy: ' + str(acc) + "\n"*2)
        f.write('Correct num: ' + str(totalCorrect) + "\n"*2)
        f.write('total sample num: ' + str(totalNum) + '\n'*2)
        f.write('sample num:\n' + str(np.array(sampleNum)) + '\n'*2)
        f.write('confusionMatrix:\n' + str(confusionMatrix.cm))

def main():
    opt = parse_args()
    thisDir = tm.get_result_dir()
    opt.thisDir = thisDir
    get_new_label(opt)
    # test(**vars(opt))
    
if __name__ == '__main__':
    main()
