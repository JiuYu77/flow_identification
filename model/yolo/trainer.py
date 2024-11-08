# -*- coding: UTF-8 -*-
import os
import torch
from torch import optim
import yaml

from model.yolo1d import *
from model.nn.yolo import YOLO1D
from utils import tu, ph, cfg, print_color, data_loader, FlowDataset, uloss, ROOT, tm, plot


class BaseTrainer:
    def __init__(
            self,
            dataset,  # 数据集的名字，和.yaml文件的文件名相同  conf/dataset/
            epochNum,  # Number of epochs to train.
            batchSize, sampleLength, step,
            transform:str,  # 用于数据预处理
            lr,  # learning rate
            shuffleFlag, numWorkers,
            modelYaml,  # yaml文件的名字, 如yolov8_1D-cls.yaml。位于conf/yolov8_1D目录, 其实放在conf目录就可以，会在conf文件夹根据文件名搜索.yaml文件
            scale,  # n s
            model,  # 模型参数文件（xxx_params.pt）的路径
            lossName,  # 损失函数
            optimName  # 优化器，优化算法，用来更新模型参数
    ) -> None:
        self.dataset = dataset
        self.epochNum = epochNum
        self.batchSize, self.sampleLength, self.step = batchSize, sampleLength, step
        self.transform = transform
        self.lr = lr
        self.shuffleFlag, self.numWorkers = shuffleFlag, numWorkers
        self.modelYaml = modelYaml
        self.scale = scale
        self.model = model
        self.lossName = lossName
        self.optimName = optimName
        self.net = None

    def train(self):
        self._setup_train()  # 训练设置
        # 路径
        thisDir = tm.get_result_dir() + f"_{self.netName}"
        resultPath = self.resultPath = os.path.join(ROOT, 'result', 'train', thisDir)
        ph.checkAndInitPath([resultPath, os.path.join(resultPath,  'weights')])
        info_fp_path = os.path.join(resultPath, 'info.yaml')  # 通过yaml文件记录模型数据
        bestWeightPath = os.path.join(resultPath, 'weights', "best_params.pt")  # 模型参数文件
        lastWeightPath = os.path.join(resultPath, 'weights', "last_params.pt")  # 模型参数文件
        epochPath = os.path.join(resultPath, "epoch")  # 训练一个epoch的数据
        trainIterPath = os.path.join(resultPath, 'train_iter')
        valIterPath = os.path.join(resultPath, 'val_iter')
        task_info = {
            "torch_version": str(torch.__version__),
            "dataset": self.dataset,
            "task_name": "Model training",
            "sample_length": self.sampleLength,
            "step": self.step,
            "transform": self.transform,
            "batch_size": self.batchSize,
            "shuffle": self.shuffle,
            "numWorkers": self.numWorkers,
            "model_name": self.netName,
            "model_parameter_amount": f"{self.modelParamAmount / 1e6:.3f}M",
            "model_settings":{'model_scale': self.net.scale, "modelYaml": self.modelYaml, 'fuse_':self.net.fuse, 'split_':self.net.split, 'initweightName': self.net.initweightName},
            "lr": self.lr,
            "epoch_num": self.epochNum,
            "epoch": self.epochNum,
            "loss_function": self.loss.__class__.__name__,
            "train_time_consuming": None,
            # "optimizer": {self.optimizer.__class__.__name__: self.optimizer.defaults},
            "optimizer": {self.optimizer.__class__.__name__: self.optimizer.state_dict()},
        }
        yaml.dump(task_info, open(info_fp_path, "w"), sort_keys=False)
    
        with open(trainIterPath, 'a+') as tIter_fp:
            tIter_fp.write(
                f"{'epoch':>6}\t{'batch':>6}\t{'NSample':>6}\t{'AccNum':>6}\t{'ACC':>6}\t{'Loss':>6}\t{'AVGLoss':>6}\n")

        with open(valIterPath, 'a+') as vIter_fp:
            vIter_fp.write(
                f"{'epoch':>6}\t{'batch':>6}\t{'NSample':>6}\t{'AccNum':>6}\t{'ACC':>6}\t{'Loss':>6}\t{'AVGLoss':>6}\n")

        # 折线图
        line = plot.Line('epoch', xlim=[1, self.epochNum], legend=['train loss', 'train acc', 'val acc'], figsize=(5, 4))
        accLine = plot.Line('epoch', 'Accuracy', xlim=[1, self.epochNum], legend=['train acc', 'val acc'], fmts=('b-', 'r-'), figsize=(5, 4))
        lossLine = plot.Line('epoch', 'Loss', xlim=[1, self.epochNum], legend=['train loss', 'val loss'], fmts=('b-', 'r-'), figsize=(5, 4))

        batchNum = len(self.trainIter)  # 训练集，一个epoch有多少个batch
        timer = tu.Timer()
        bestAcc = 0
        print("-----------------------------------------")
        print(f"|\033[33m Start training:\033[0m")
        print(f"|\033[32m training device:\033[0m {self.device}")
        print(f"| epoch_num {self.epochNum}, lr {self.lr}, batch_size {self.batchSize}")
        print(f"| \033[34mmodel_scale:\033[0m {self.net.scale}")
        print(f"| train_batch_num: {batchNum}")
        print(f"| val_batch_num: {len(self.valIter)}")
        print("-----------------------------------------")

        print_color(["bright_green", "bold", "preparing data..."])
        for epoch in range(self.epochNum):
            epoch_ = self.epoch = epoch + 1
            # 训练
            accumulatorTrain = tu.Accumulator(3)
            self.net.train()
            for i, (X, y) in enumerate(self.trainIter):
                timer.start()
                # 正向传播
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                loss_ = self.loss(y_hat, y)

                # 反向传播，更新参数
                loss_.backward()
                self.optimizer.step()

                sampleNum = len(y)  # 一个batch的样本数
                correctNum = tu.accuracy(y_hat, y)
                train_loss = loss_.item()  # 平均到一个样本上的损失
                batchLoss = train_loss * sampleNum
                # accumulatorTrain.add(平均损失 * 一个batch的样本数, 预测正确的数量, 一个batch的样本数)
                if type(self.optimizer) == optim.SGD:  # 如果优化算法是随机梯度下降
                    with torch.no_grad():
                        accumulatorTrain.add(batchLoss, correctNum, sampleNum)
                elif type(self.optimizer) == optim.Adam:
                    accumulatorTrain.add(batchLoss, correctNum, sampleNum)
                else:
                    accumulatorTrain.add(batchLoss, correctNum, sampleNum)
                timer.stop()

                prg = f"{int((i + 1) / batchNum * 100)}%"  # 进度，百分比
                print(f"\r\033[K\033[31mepoch\033[0m {epoch_:>3}/{self.epochNum}    \033[31mbatch:\033[0m{i+1}/{batchNum}    \033[31mprogress:\033[0m{prg}    \033[31msample_num:\033[0m{sampleNum}    \033[31mtrain_loss:\033[0m{train_loss:.5f}",end='\r')
                # 训练数据记录
                batchAcc = correctNum / sampleNum
                with open(trainIterPath, 'a+') as tIter_fp:
                    tIter_fp.write(
                        f"{epoch:>6}\t{i+1:>6}\t{sampleNum:>6}\t{correctNum:>6}\t{batchAcc:>6.4}\t{batchLoss:>6.2f}\t{train_loss:>6.3f}\n")

            # 一个epoch的 训练损失 和 准确率
            trainLoss = accumulatorTrain[0] / accumulatorTrain[2]  # 每个样本 平均损失
            trainAcc = accumulatorTrain[1] / accumulatorTrain[2]  # 每个样本上的 平均准确率

            # ----------- 验证 ----------
            if type(self.optimizer) == optim.SGD:  # 如果优化算法是随机梯度下降
                with torch.no_grad():
                    valLoss, valAcc = self.val()
            elif type(self.optimizer) == optim.Adam:
                valLoss, valAcc = self.val()
            else:
                valLoss, valAcc = self.val()
            # ----------- 验证 ----------

            # 保存模型
            bestAccSymbol = ''
            if bestAcc < valAcc:
                bestAcc = valAcc
                self.net.save(bestWeightPath)  # 保存网络参数
                bestAccSymbol = '  *'
            if epoch + 1 == self.epochNum:
                self.net.save(lastWeightPath)  # 保存网络参数

            # 打印一个epoch的信息
            print(f"\033[35mepoch\033[0m {epoch_:>3}/{self.epochNum}    \033[32mtrain_loss:\033[0m{trainLoss:.5f}    \033[32mtrain_acc:\033[0m{trainAcc:.5f}"\
                f"    \033[36mval_loss:\033[0m{valLoss:.5f}    \033[36mval_acc:\033[0m{valAcc:.5f}\033[31m{bestAccSymbol}\033[0m")

            # 保存一个epoch的信息
            with open(epochPath, 'a+') as f:
                f.write(f"epoch {epoch_}    train_loss: {trainLoss:.5f}    train_acc: {trainAcc:.5f}    val_loss: {valLoss:.5f}    val_acc: {valAcc:.5f}{bestAccSymbol}\n")

            # 记录画图用的数据：一个epoch的训练平均损失、训练准确率、验证准确率、验证平均损失
            line.add(epoch + 1, [trainLoss, trainAcc, valAcc])
            accLine.add(epoch + 1, [trainAcc, valAcc])
            lossLine.add(epoch + 1, [trainLoss, valLoss])
    
        # 画图
        print_color(["drawing..."])
        img_lossAccPath = os.path.join(resultPath, 'loss_acc.png')
        img_accPath = os.path.join(resultPath, 'acc.png')
        img_lossPath = os.path.join(resultPath, 'loss.png')
        line.Init()
        line.draw_save(img_lossAccPath)
        accLine.Init()
        accLine.draw_save(img_accPath)
        lossLine.Init()
        lossLine.draw_save(img_lossPath)

        # 训练结束（完成）
        self.trainTimer.stop()
        timeConsuming = tm.sec_to_HMS(self.trainTimer.sum())
        task_info["train_time_consuming"] = timeConsuming
        yaml.dump(task_info, open(info_fp_path, "w"), sort_keys=False)

        print("----------------------------------------------")
        print(f"|\033[33m End training:\033[0m") 
        print(f"| loss {trainLoss:.4f}, train_acc {trainAcc:.4f}, val_acc {valAcc:.4f}")
        print(f"| {accumulatorTrain[2] * self.epochNum / timer.sum():.1f} samlpes/sec, on \033[33m{str(self.device)}\033[0m")  # 每秒处理的样本数, 使用的设备
        print(f"| train time consuming: {timeConsuming}")  # 训练耗时
        print("----------------------------------------------")

    def _setup_train(self):
        shuffleFlag, dataset, sampleLength, step, transform, batchSize, numWorkers = \
                                        self.shuffleFlag, self.dataset, self.sampleLength, self.step, self.transform, self.batchSize, self.numWorkers
        assert shuffleFlag == 1 or shuffleFlag == 0, f'shuffle_flag ValueError, except 0 or 1, but got {shuffleFlag}'
        self.trainTimer = tu.Timer()
        self.trainTimer.start()
        self.shuffle = shuffle = shuffleFlag == 1 or not shuffleFlag == 0
        self.device = tu.get_device()

        # 数据集
        deviceName = tu.getDeviceName()
        if deviceName == "windows":
            numWorkers = 0
        train_val_PathList, classNum, self.data = cfg.get_dataset_info(dataset, deviceName)
        trainDatasetPath = train_val_PathList[0]
        valDatasetPath = train_val_PathList[1]
        # 训练 数据加载器
        self.trainIter = data_loader(FlowDataset, trainDatasetPath, sampleLength, step, transform, batchSize, shuffle=shuffle, numWorkers=numWorkers)
        if transform.lower().find('noise') != -1:
            transform = 'standardization_zScore'
        # 验证 数据加载器
        self.valIter = data_loader(FlowDataset, valDatasetPath, sampleLength, step, transform, batchSize, shuffle=shuffle, numWorkers=numWorkers)

        self.loss = uloss.smart_lossFunction(self.lossName)  # 损失函数
        # loss = uloss.smart_lossFunction('FocalLoss', classNum)  # 损失函数
        # loss = uloss.smart_lossFunction('FocalLoss', class_num=classNum)  # 损失函数
        self.get_net()  # net
        self.optimizer = tu.smart_optimizer(self.net, self.optimName, self.lr)  # 优化器
        self.netName = self.net.__class__.__name__
        self.modelParamAmount = sum([p.nelement() for p in self.net.parameters()])

    def get_net(self):
        # 网络
        print_color(["bright_green", "bold", "\nloading model..."])
        fuse_, split_, initweightName = False, False, 'xavier'
        if self.model is not None:  # 在已有模型的基础上继续训练
            # 读取训练信息
            path = os.path.dirname(os.path.dirname(self.model))
            yml = cfg.yaml_load(os.path.join(path, 'info.yaml'))
            fuse_, split_ = yml['model_settings']['fuse_'], yml['model_settings']['split_']
            self.scale = yml['model_settings']['model_scale']
            self.modelYaml = yml['model_settings']['modelYaml']
        self.net = YOLO1D(self.modelYaml, self.model, fuse_=fuse_, split_=split_, scale=self.scale, initweightName=initweightName, device=self.device)  # 实例化模型
        self.net.set_names(self.data['names'])

    def val(self):
        '''validate'''
        valLoss, valAcc = tu.val(self.net, self.valIter, self.device, self.loss, self.epoch, self.epochNum, self.resultPath)
        return valLoss, valAcc

    @staticmethod
    def train_function(
            dataset='v4_press4',
            epochNum=100,
            batchSize=64, sampleLength=4096, step=2048,
            transform:str='standardization_zScore',  # 用于数据预处理
            lr=0.00001,
            shuffleFlag=1, numWorkers=4,
            modelYaml=MODEL_YAML_DEFAULT,  # yaml文件的名字, 如yolov8_1D-cls.yaml。位于conf/yolov8_1D目录, 其实放在conf目录就可以，会在conf文件夹根据文件名搜索.yaml文件
            scale='n',
            model=None,
            lossName='CrossEntropyLoss',
            optimName='SGD'  # 优化器，优化算法，用来更新模型参数
    ):
        '''
        使用方法：直接调用并传参数, BaseTrainer.train_function(...)
        '''
        assert shuffleFlag == 1 or shuffleFlag == 0, f'shuffle_flag ValueError, except 0 or 1, but got {shuffleFlag}'
        trainTimer = tu.Timer()
        trainTimer.start()
        shuffle = shuffleFlag == 1 or not shuffleFlag == 0
        device = tu.get_device()

        # 数据集
        print_color(["loading dataset..."])
        deviceName = tu.getDeviceName()
        if deviceName == "windows":
            numWorkers = 0
        train_val_PathList, classNum = cfg.get_dataset_info(dataset, deviceName)
        trainDatasetPath = train_val_PathList[0]
        valDatasetPath = train_val_PathList[1]
        trainIter = data_loader(FlowDataset, trainDatasetPath, sampleLength, step, transform, batchSize, shuffle=shuffle, numWorkers=numWorkers)
        if transform.lower().find('noise') != -1:
            transform = 'standardization_zScore'
        valIter = data_loader(FlowDataset, valDatasetPath, sampleLength, step, transform, batchSize, shuffle=shuffle, numWorkers=numWorkers)

        # 模型
        print_color(["bright_green", "bold", "\nloading model..."])
        fuse_, split_, initweightName = False, False, 'xavier'
        if model is not None:  # 在已有模型的基础上继续训练
            # 读取训练信息
            path = os.path.dirname(model)
            yml = cfg.yaml_load(os.path.join(path, 'info.yaml'))
            fuse_, split_ = yml['model_settings']['fuse_'], yml['model_settings']['split_']
            scale = yml['model_settings']['model_scale']
        net = YOLOv8_1D(modelYaml, model, fuse_=fuse_, split_=split_, scale=scale, initweightName=initweightName, device=device)  # 实例化模型

        loss = uloss.smart_lossFunction(lossName)  # 损失函数
        # loss = uloss.smart_lossFunction('FocalLoss', classNum)  # 损失函数
        # loss = uloss.smart_lossFunction('FocalLoss', class_num=classNum)  # 损失函数
        optimizer = tu.smart_optimizer(net, optimName, lr)  # 优化器
        netName = net.__class__.__name__
        modelParamAmount = sum([p.nelement() for p in net.parameters()])

        # 路径
        thisDir = tm.get_result_dir() + f"_{netName}"
        resultPath = os.path.join(ROOT, 'result', 'train', thisDir)
        ph.checkAndInitPath(resultPath)
        info_fp_path = os.path.join(resultPath, 'info.yaml')  # 通过yaml文件记录模型数据
        bestWeightPath = os.path.join(resultPath, "best_params.pt")  # 模型参数文件
        lastWeightPath = os.path.join(resultPath, "last_params.pt")  # 模型参数文件
        epochPath = os.path.join(resultPath, "epoch")  # 训练一个epoch的数据
        trainIterPath = os.path.join(resultPath, 'train_iter')
        valIterPath = os.path.join(resultPath, 'val_iter')
        task_info = {
            "torch_version": str(torch.__version__),
            "dataset": dataset,
            "task_name": "Model training",
            "sample_length": sampleLength,
            "step": step,
            "transform": transform,
            "batch_size": batchSize,
            "shuffle": shuffle,
            "numWorkers": numWorkers,
            "model_name": netName,
            "model_parameter_amount": f"{modelParamAmount / 1e6:.3f}M",
            "model_settings":{'model_scale': net.scale, "modelYaml": modelYaml, 'fuse_':fuse_, 'split_':split_, 'initweightName': initweightName},
            "lr": lr,
            "epoch_num": epochNum,
            "loss_function": loss.__class__.__name__,
            "train_time_consuming": None,
            # "optimizer": {optimizer.__class__.__name__: optimizer.defaults},
            "optimizer": {optimizer.__class__.__name__: optimizer.state_dict()},
        }
        yaml.dump(task_info, open(info_fp_path, "w"), sort_keys=False)
    
        with open(trainIterPath, 'a+') as tIter_fp:
            tIter_fp.write(
                f"{'epoch':>6}\t{'batch':>6}\t{'NSample':>6}\t{'AccNum':>6}\t{'ACC':>6}\t{'Loss':>6}\t{'AVGLoss':>6}\n")

        with open(valIterPath, 'a+') as vIter_fp:
            vIter_fp.write(
                f"{'epoch':>6}\t{'batch':>6}\t{'NSample':>6}\t{'AccNum':>6}\t{'ACC':>6}\t{'Loss':>6}\t{'AVGLoss':>6}\n")

        # 折线图
        line = plot.Line('epoch', xlim=[1, epochNum], legend=['train loss', 'train acc', 'val acc'], figsize=(5, 4))
        accLine = plot.Line('epoch', 'Accuracy', xlim=[1, epochNum], legend=['train acc', 'val acc'], fmts=('b-', 'r-'), figsize=(5, 4))
        lossLine = plot.Line('epoch', 'Loss', xlim=[1, epochNum], legend=['train loss', 'val loss'], fmts=('b-', 'r-'), figsize=(5, 4))

        batchNum = len(trainIter)  # 训练集，一个epoch有多少个batch
        timer = tu.Timer()
        bestAcc = 0
        print("-----------------------------------------")
        print(f"|\033[33m Start training:\033[0m")
        print(f"|\033[32m training device:\033[0m {device}")
        print(f"| epoch_num {epochNum}, lr {lr}, batch_size {batchSize}")
        print(f"| \033[34mmodel_scale:\033[0m {net.scale}")
        print(f"| train_batch_num: {batchNum}")
        print(f"| val_batch_num: {len(valIter)}")
        print("-----------------------------------------")
        epoch_ = 0
        print_color(["bright_green", "bold", "preparing data..."])
        for epoch in range(epochNum):
            epoch_ = epoch + 1
            # 训练
            accumulatorTrain = tu.Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(trainIter):
                timer.start()
                # 正向传播
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                loss_ = loss(y_hat, y)

                # 反向传播，更新参数
                loss_.backward()
                optimizer.step()

                sampleNum = len(y)  # 一个batch的样本数
                correctNum = tu.accuracy(y_hat, y)
                train_loss = loss_.item()  # 平均到一个样本上的损失
                batchLoss = train_loss * sampleNum
                # accumulatorTrain.add(平均损失 * 一个batch的样本数, 预测正确的数量, 一个batch的样本数)
                if type(optimizer) == optim.SGD:  # 如果优化算法是随机梯度下降
                    with torch.no_grad():
                        accumulatorTrain.add(batchLoss, correctNum, sampleNum)
                elif type(optimizer) == optim.Adam:
                    accumulatorTrain.add(batchLoss, correctNum, sampleNum)
                else:
                    accumulatorTrain.add(batchLoss, correctNum, sampleNum)
                timer.stop()

                prg = f"{int((i + 1) / batchNum * 100)}%"  # 进度，百分比
                print(f"\r\033[K\033[31mepoch\033[0m {epoch_:>3}/{epochNum}    \033[31mbatch:\033[0m{i}    \033[31mprogress:\033[0m{prg}    \033[31msample_num:\033[0m{sampleNum}    \033[31mtrain_loss:\033[0m{train_loss:.5f}",end='\r')

                # 训练数据记录
                batchAcc = correctNum / sampleNum
                with open(trainIterPath, 'a+') as tIter_fp:
                    tIter_fp.write(
                        f"{epoch:>6}\t{i+1:>6}\t{sampleNum:>6}\t{correctNum:>6}\t{batchAcc:>6.4}\t{batchLoss:>6.2f}\t{train_loss:>6.3f}\n")

            # 一个epoch的 训练损失 和 准确率
            trainLoss = accumulatorTrain[0] / accumulatorTrain[2]  # 每个样本 平均损失
            trainAcc = accumulatorTrain[1] / accumulatorTrain[2]  # 每个样本上的 平均准确率

            # ----------- 验证 ----------
            if type(optimizer) == optim.SGD:  # 如果优化算法是随机梯度下降
                with torch.no_grad():
                    valLoss, valAcc = tu.val(net, valIter, device, loss, epoch_, epochNum, resultPath)
            elif type(optimizer) == optim.Adam:
                valLoss, valAcc = tu.val(net, valIter, device, loss, epoch_, epochNum, resultPath)
            else:
                valLoss, valAcc = tu.val(net, valIter, device, loss, epoch_, epochNum, resultPath)
            # ----------- 验证 ----------

            # 保存模型
            bestAccSymbol = ''
            if bestAcc < valAcc:
                bestAcc = valAcc
                net.save(bestWeightPath)  # 保存网络参数
                bestAccSymbol = '  *'
            if epoch + 1 == epochNum:
                net.save(lastWeightPath)  # 保存网络参数

            # 打印一个epoch的信息
            print(f"\033[35mepoch\033[0m {epoch_:>3}/{epochNum}    \033[32mtrain_loss:\033[0m{trainLoss:.5f}    \033[32mtrain_acc:\033[0m{trainAcc:.5f}"\
                f"    \033[36mval_loss:\033[0m{valLoss:.5f}    \033[36mval_acc:\033[0m{valAcc:.5f}\033[31m{bestAccSymbol}\033[0m")

            # 保存一个epoch的信息
            with open(epochPath, 'a+') as f:
                f.write(f"epoch {epoch_}    train_loss: {trainLoss:.5f}    train_acc: {trainAcc:.5f}    val_loss: {valLoss:.5f}    val_acc: {valAcc:.5f}{bestAccSymbol}\n")

            # 记录画图用的数据：一个epoch的训练平均损失、训练准确率、验证准确率、验证平均损失
            line.add(epoch + 1, [trainLoss, trainAcc, valAcc])
            accLine.add(epoch + 1, [trainAcc, valAcc])
            lossLine.add(epoch + 1, [trainLoss, valLoss])
    
        # 画图
        print_color(["drawing..."])
        img_lossAccPath = os.path.join(resultPath, 'loss_acc.png')
        img_accPath = os.path.join(resultPath, 'acc.png')
        img_lossPath = os.path.join(resultPath, 'loss.png')
        line.Init()
        line.draw_save(img_lossAccPath)
        accLine.Init()
        accLine.draw_save(img_accPath)
        lossLine.Init()
        lossLine.draw_save(img_lossPath)

        # 训练结束（完成）
        trainTimer.stop()
        timeConsuming = tm.sec_to_HMS(trainTimer.sum())
        task_info["train_time_consuming"] = timeConsuming
        yaml.dump(task_info, open(info_fp_path, "w"), sort_keys=False)

        print("----------------------------------------------")
        print(f"|\033[33m End training:\033[0m") 
        print(f"| loss {trainLoss:.4f}, train_acc {trainAcc:.4f}, val_acc {valAcc:.4f}")
        print(f"| {accumulatorTrain[2] * epochNum / timer.sum():.1f} samlpes/sec, on \033[33m{str(device)}\033[0m")  # 每秒处理的样本数, 使用的设备
        print(f"| train time consuming: {timeConsuming}")  # 训练耗时
        print("----------------------------------------------")