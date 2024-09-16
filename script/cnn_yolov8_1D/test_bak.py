# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import sys
sys.path.append('.')
from model import YOLOv8_1D, MODEL_YAML_DEFAULT
from utils import FlowDataset, data_loader, tu, cfg, colorstr, print_color, ROOT, ph, tm, plot
import os
import torch
import numpy as np
import yaml


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['v4_press4', 'v4_wms', 'v1_wms'], default='v4_press4', help="数据集的名字，和yaml配置文件名相同，不包括.yaml后缀")
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

    # netPath = os.path.join(os.getcwd(), 'result', 'train', '20240317.163818_Yolov8_1D', 'best_params.pt')
    # netPath = os.path.join('result', 'train', '20240318.132209_Yolov8_1D', 'best_params.pt')
    # netPath = os.path.join('result', 'train', '20240327.090238_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240327.203215_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240412.162548_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240530.154012_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240704.195356_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240705.110251_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240705.140052_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240717.220859_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240718.085919_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240718.105346_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240718.150219_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240718.152357_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240718.201457_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240719.113718_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240719.150006_Yolov8_1D', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20240720.142313_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240718.105346_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240721.123959_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240721.161448_Yolov8_1D', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20240722.112911_Yolov8_1D', 'best_params.pt')
    weight = netPath
    parser.add_argument('-w', '--weights', type=str, default=weight)

    opt = parser.parse_args()
    return opt

def test(dataset,
         batchSize,
         sampleLength,
         step,
         transform:str,
         shuffleFlag,
         numWorkers,
         modelYaml,
         weights
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
    print_color(['loading test dataset...'])
    deviceName = tu.getDeviceName()
    if deviceName == "windows":
        numWorkers = 0
    testDatasetPath, classNum = cfg.get_dataset_info(dataset, deviceName, train=False)
    testIter = data_loader(FlowDataset, testDatasetPath, sampleLength, step, transform, batchSize, shuffle=shuffle, numWorkers=numWorkers)

    # 模型
    print('loading model...')
    scale = yml['model_settings']['model_scale']
    # modelYaml = modelYaml if modelYaml else yml['model_settings']['modelYaml']
    fuse_, split_ = yml['model_settings']['fuse_'], yml['model_settings']['split_']
    net = YOLOv8_1D(modelYaml, weights, scale=scale, fuse_=fuse_, split_=split_, device=device)
    net.eval()
    netName = net.__class__.__name__
    modelParamAmount = sum([p.nelement() for p in net.parameters()])

    # 路径
    thisDir = tm.get_result_dir() + f"_{netName}"
    resultPath = os.path.join(ROOT, 'result', 'test', thisDir)
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
        "speed": None
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
    test(**vars(opt))
    
if __name__ == '__main__':
    main()
