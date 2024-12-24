# -*- coding: UTF-8 -*-
import os
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

from jyu.engine.trainer import BaseTrainer
from jyu.utils import tu, data_loader, FlowDataset, cfg, uloss, print_color, tm, ROOT, ph, plot

class Trainer(BaseTrainer):
    def data_loader(self, Dataset, datasetPath, sampleLength:int, step:int, transform, clss=None, supervised=True,
                batchSize:int=64, shuffle=True, numWorkers=0, PseudoLabel=None):
        self.dataset_obj = Dataset(datasetPath, sampleLength, step, transform, clss, supervised)
        if PseudoLabel:
            self.dataset_obj.allLabel = PseudoLabel
        dataloader = DataLoader(self.dataset_obj, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers)
        return dataloader

    def pseudo_label(self, trainDatasetPath, sampleLength, step, transform):
        import numpy as np
        from jyu.tml import do_pca, do_k_means
        PseudoLabel = []
        dataset = FlowDataset(trainDatasetPath, sampleLength, step, transform, clss=-1)
        print_color(['generate pseudo label...'])
        dataset.allSample = np.array(dataset.allSample)
        dataset.do_transform()
        data = dataset.allSample

        pca_data = do_pca(data, 3, True)
        X = low_dim_data = pca_data

        pre = do_k_means(X, 7, 'auto')
        PseudoLabel = [int(item) for item in pre.tolist()]
        print(PseudoLabel)
        return PseudoLabel
    
    def do_pseudo_label(self, X, index):
        import numpy as np
        from jyu.tml import do_pca, do_k_means, do_dbscan
        PseudoLabel = []
        print_color(['generate pseudo label...'])
        data = X.squeeze(1)

        pca_data = do_pca(data, 2, True)
        X = low_dim_data = pca_data

        # pre = do_k_means(X, 7, 'auto')
        pre = do_dbscan(X, 1, 10)
        m = pre.argmax(axis=1)
        pre[pre==-1] = m
        # key = np.unique(pre)
        # results = {}
        # for k in key:
        #     v =  pre[ pre == k ].size
        #     results[k] = v

        PseudoLabel = [int(item) for item in pre.tolist()]
        print(PseudoLabel)
        for i, v in enumerate(PseudoLabel):
            idx = int(index[i])
            self.dataset_obj.allLabel[idx] = int(v)

        return PseudoLabel

    def get_porb(self, y_hat_softmax):
        ys_max = y_hat_softmax.max()
        ys_min = y_hat_softmax.min()
        prob = (ys_max + ys_min) / 2
        return prob

    def update_pseudo_label(self, y_hat, index, prob=0.5):
        preLabel = y_hat.argmax(axis=1)

        for i in range(0, y_hat.shape[0]):
            y_hat[i] = y_hat[i].softmax(dim=0)
        tmp = y_hat.max(axis=1)
        y_hat_softmax = tmp.values
        # y_hat_softmax = tmp.values.softmax(dim=0)  # y_hat_softmax = torch.softmax(tmp.values, dim=0)
        a ='aaa'
        prob = self.get_porb(y_hat_softmax)
        # prob = 0.1
        for i, v in enumerate(preLabel):
            if y_hat_softmax[i] > prob:
                idx = int(index[i])
                # self.trainIter.dataset.allLabel[idx] = v
                self.dataset_obj.allLabel[idx] = int(v)

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

        # 生成初始伪标签
        labels = self.pseudo_label(trainDatasetPath, sampleLength, step, "zScore_std")

        # 训练 数据加载器
        self.trainIter = self.data_loader(FlowDataset, trainDatasetPath, sampleLength, step, transform, clss=-1, supervised=False,
                                     batchSize=batchSize, shuffle=shuffle, numWorkers=numWorkers, PseudoLabel=labels)
        if transform.lower().find('noise') != -1:
            transform = 'zScore_std'
        # 验证 数据加载器

        self.loss = uloss.smart_lossFunction(self.lossName)  # 损失函数
        # loss = uloss.smart_lossFunction('FocalLoss', classNum)  # 损失函数
        # loss = uloss.smart_lossFunction('FocalLoss', class_num=classNum)  # 损失函数
        self.get_net()  # net
        self.optimizer = tu.smart_optimizer(self.net, self.optimName, self.lr)  # 优化器
        self.netName = self.net.__class__.__name__
        self.modelParamAmount = sum([p.nelement() for p in self.net.parameters()])

    def train(self):
        self._setup_train()  # 训练设置
        # 路径
        thisDir = tm.get_result_dir() + f"_{self.netName}"
        resultPath = self.resultPath = os.path.join(ROOT, 'result', 'train_unsupervised', thisDir)
        ph.checkAndInitPath([resultPath, os.path.join(resultPath,  'weights')])
        info_fp_path = os.path.join(resultPath, 'info.yaml')  # 通过yaml文件记录模型数据
        bestWeightPath = os.path.join(resultPath, 'weights', "best_params.pt")  # 模型参数文件
        lastWeightPath = os.path.join(resultPath, 'weights', "last_params.pt")  # 模型参数文件
        epochPath = os.path.join(resultPath, "epoch")  # 训练一个epoch的数据
        trainIterPath = os.path.join(resultPath, 'train_iter')
        # valIterPath = os.path.join(resultPath, 'val_iter')
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
            "model_settings":{'model_scale': self.net.scale, "modelYaml": self.modelYaml, 'fuse':self.net.fuse, 'split':self.net.split, 'initweightName': self.net.initweightName},
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

        # with open(valIterPath, 'a+') as vIter_fp:
        #     vIter_fp.write(
        #         f"{'epoch':>6}\t{'batch':>6}\t{'NSample':>6}\t{'AccNum':>6}\t{'ACC':>6}\t{'Loss':>6}\t{'AVGLoss':>6}\n")

        # 折线图
        # line = plot.Line('epoch', xlim=[1, self.epochNum], legend=['train loss', 'train acc', 'val acc'], figsize=(5, 4))
        # accLine = plot.Line('epoch', 'Accuracy', xlim=[1, self.epochNum], legend=['train acc', 'val acc'], fmts=('b-', 'r-'), figsize=(5, 4))
        # lossLine = plot.Line('epoch', 'Loss', xlim=[1, self.epochNum], legend=['train loss', 'val loss'], fmts=('b-', 'r-'), figsize=(5, 4))
        line = plot.Line('epoch', xlim=[1, self.epochNum], legend=['train loss', 'train acc'], figsize=(5, 4))
        accLine = plot.Line('epoch', 'Accuracy', xlim=[1, self.epochNum], legend=['train acc'], fmts=('b-', 'r-'), figsize=(5, 4))
        lossLine = plot.Line('epoch', 'Loss', xlim=[1, self.epochNum], legend=['train loss'], fmts=('b-', 'r-'), figsize=(5, 4))

        batchNum = len(self.trainIter)  # 训练集，一个epoch有多少个batch
        timer = tu.Timer()
        bestAcc = 0
        print("-----------------------------------------")
        print(f"|\033[33m Start training:\033[0m")
        print(f"|\033[32m training device:\033[0m {self.device}")
        print(f"| epoch_num {self.epochNum}, lr {self.lr}, batch_size {self.batchSize}")
        print(f"| \033[34mmodel_scale:\033[0m {self.net.scale}")
        print(f"| train_batch_num: {batchNum}")
        # print(f"| val_batch_num: {len(self.valIter)}")
        print("-----------------------------------------")

        print_color(["bright_green", "bold", "preparing data..."])
        for epoch in range(self.epochNum):
            epoch_ = self.epoch = epoch + 1
            # 训练
            accumulatorTrain = tu.Accumulator(3)
            self.net.train()
            for i, (X, y, index) in enumerate(self.trainIter):
                if epoch == 0:
                    tt = y.dtype
                    y = self.do_pseudo_label(X, index)
                    y = torch.tensor(y, dtype=tt)
                timer.start()
                # 正向传播
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                loss_ = self.loss(y_hat, y)

                # 反向传播，更新参数
                loss_.backward()
                self.optimizer.step()

                # self.update_pseudo_label(y_hat, index) # 更新伪标签

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
            # if type(self.optimizer) == optim.SGD:  # 如果优化算法是随机梯度下降
            #     with torch.no_grad():
            #         valLoss, valAcc = self.val()
            # elif type(self.optimizer) == optim.Adam:
            #     valLoss, valAcc = self.val()
            # else:
            #     valLoss, valAcc = self.val()
            # ----------- 验证 ----------

            # 保存模型
            bestAccSymbol = ''
            # if bestAcc < valAcc:
            if bestAcc < trainAcc:
                # bestAcc = valAcc
                bestAcc = trainAcc
                self.net.save(bestWeightPath)  # 保存网络参数
                bestAccSymbol = '  *'
            if epoch + 1 == self.epochNum:
                self.net.save(lastWeightPath)  # 保存网络参数

            # 打印一个epoch的信息
            # print(f"\033[35mepoch\033[0m {epoch_:>3}/{self.epochNum}    \033[32mtrain_loss:\033[0m{trainLoss:.5f}    \033[32mtrain_acc:\033[0m{trainAcc:.5f}"\
            #     f"    \033[36mval_loss:\033[0m{valLoss:.5f}    \033[36mval_acc:\033[0m{valAcc:.5f}\033[31m{bestAccSymbol}\033[0m")
            print(f"\033[K\033[35mepoch\033[0m {epoch_:>3}/{self.epochNum}    \033[32mtrain_loss:\033[0m{trainLoss:.5f}    \033[32mtrain_acc:\033[0m{trainAcc:.5f}"\
                f"  \033[31m{bestAccSymbol}\033[0m")

            # 保存一个epoch的信息
            with open(epochPath, 'a+') as f:
                # f.write(f"epoch {epoch_}    train_loss: {trainLoss:.5f}    train_acc: {trainAcc:.5f}    val_loss: {valLoss:.5f}    val_acc: {valAcc:.5f}{bestAccSymbol}\n")
                f.write(f"epoch {epoch_}    train_loss: {trainLoss:.5f}    train_acc: {trainAcc:.5f}    {bestAccSymbol}\n")

            # 记录画图用的数据：一个epoch的训练平均损失、训练准确率、验证准确率、验证平均损失
            # line.add(epoch + 1, [trainLoss, trainAcc, valAcc])
            # accLine.add(epoch + 1, [trainAcc, valAcc])
            # lossLine.add(epoch + 1, [trainLoss, valLoss])
            line.add(epoch + 1, [trainLoss, trainAcc])
            accLine.add(epoch + 1, [trainAcc])
            lossLine.add(epoch + 1, [trainLoss])

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
        # print(f"| loss {trainLoss:.4f}, train_acc {trainAcc:.4f}, val_acc {valAcc:.4f}")
        print(f"| loss {trainLoss:.4f}, train_acc {trainAcc:.4f}")
        print(f"| {accumulatorTrain[2] * self.epochNum / timer.sum():.1f} samlpes/sec, on \033[33m{str(self.device)}\033[0m")  # 每秒处理的样本数, 使用的设备
        print(f"| train time consuming: {timeConsuming}")  # 训练耗时
        print("----------------------------------------------")
