# -*- coding: UTF-8 -*-
import os
import torch
import yaml
import numpy as np

from model.yolo1d import *
from utils import tu, ph, cfg, print_color, data_loader, FlowDataset, ROOT, tm, plot, colorstr


class BaseTester:
    def __init__(
            self,
            dataset,
            batchSize,
            sampleLength,
            step,
            transform:str,
            shuffleFlag,
            numWorkers,
            modelYaml,
            weights
    ) -> None:
        self.dataset = dataset
        self.batchSize = batchSize
        self.sampleLength = sampleLength
        self.step = step
        self.transform = transform
        self.shuffleFlag = shuffleFlag
        self.numWorkers = numWorkers
        self.modelYaml = modelYaml
        self.weights = weights

    def test(self):
        self._setup_test()
        self.net.eval()
        netName = self.net.__class__.__name__
        modelParamAmount = sum([p.nelement() for p in self.net.parameters()])

        # 路径
        thisDir = tm.get_result_dir() + f"_{netName}"
        resultPath = os.path.join(ROOT, 'result', 'test', thisDir)
        ph.checkAndInitPath(resultPath)
        # cmPath = os.path.join(resultPath, 'confusionMatrix.txt')
        cmPath = os.path.join(resultPath, 'confusionMatrix.yaml')
        confusionMatrixPath = os.path.join(resultPath, 'confusionMatrix.png')
        info_fp_path = os.path.join(resultPath, 'info.yaml')

        task_info = {
            "dataset": self.dataset,
            "task_name": "Model training",
            "model_name": netName,
            "model_scale": self.net.scale,
            "batch_size": self.batchSize,
            "sample_length": self.sampleLength,
            "step": self.step,
            "model_parameter_amount": f"{modelParamAmount / 1e6:.3f}M",
            "transform": self.transform,
            "shuffle": self.shuffle,
            "numWorkers": self.numWorkers,
            "net_path": self.weights,
            "test_time_consuming": None,
            "speed": None
        }
        yaml.dump(task_info, open(info_fp_path, "w"))

        totalCorrect = 0
        totalNum = 0
        classes, classesFlag = cfg.get_classes(self.dataset)  # 类别list, 类别对应的整数标签list
        fontsize = 12 if len(classesFlag) == 7 else 15  # 7种类别 12, 4种类别 15
        confusionMatrix = plot.ConfusionMatrix(classesFlag, normalize=True, figsize=(6, 5), fontdict=dict(fontsize=fontsize))  # fontsize=12 15
        sampleNum = [[i for i in range(self.classNum)],[0 for _ in range(self.classNum)]]  # 每种流型的样本数
        print("-----------------------------------------")
        print(f"|{colorstr('yellow', ' Start testing:')}")
        print(f"|{colorstr('green', ' testing device:')} {self.device}")
        print("-----------------------------------------")
        testTimer = tu.Timer()
        print(f"{colorstr('blue', 'timer start...')}")
        testTimer.start()

        print_color(["bright_green", "preparing data..."])
        for i, (X,y) in enumerate(self.testIter):
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.net(X)
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

    def _setup_test(self):
        shuffleFlag = self.shuffleFlag
        assert shuffleFlag == 1 or shuffleFlag == 0, f'shuffle_flag ValueError, except 0 or 1, but got {shuffleFlag}'
        self.shuffle = shuffleFlag == 1 or not shuffleFlag == 0
        self.device = tu.get_device()

        # 读取训练信息
        path = os.path.dirname(os.path.dirname(self.weights))
        yml = cfg.yaml_load(os.path.join(path, 'info.yaml'))
        # transform
        self.transform = yml['transform'] if self.transform is None else self.transform
        if self.transform.lower().find('noise') != -1:
            self.transform = 'standardization_zScore'

        # 数据集
        print_color(['loading test dataset...'])
        deviceName = tu.getDeviceName()
        if deviceName == "windows":
            self.numWorkers = 0
        testDatasetPath, self.classNum = cfg.get_dataset_info(self.dataset, deviceName, train=False)
        self.testIter = data_loader(FlowDataset, testDatasetPath, self.sampleLength, self.step, self.transform, self.batchSize, shuffle=self.shuffle, numWorkers=self.numWorkers)

        # 模型
        print('loading model...')
        scale = yml['model_settings']['model_scale']
        modelYaml = self.modelYaml if self.modelYaml else yml['model_settings']['modelYaml']  # ########
        fuse_, split_ = yml['model_settings']['fuse_'], yml['model_settings']['split_']
        self.net = YOLOv8_1D(modelYaml, self.weights, scale=scale, fuse_=fuse_, split_=split_, device=self.device)

    @staticmethod
    def test_function(
            weights,
            dataset='v4_press4',  # 数据集的名字，和yaml配置文件名相同，不包括.yaml后缀
            batchSize=64,  # Number of batch size to test.
            sampleLength=4096,
            step=2048,
            transform:str=None,
            shuffleFlag=1,
            numWorkers=4,
            modelYaml=MODEL_YAML_DEFAULT,
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