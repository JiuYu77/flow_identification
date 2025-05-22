# -*- coding: UTF-8 -*-
from jyu.engine.trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(
            self,
            dataset,  # 数据集的名字，和.yaml文件的文件名相同  conf/dataset/
            epochNum,  # Number of epochs to train.
            batchSize, sampleLength, step,
            transform:str,  # 用于数据预处理，训练集
            transform2:str,  # 用于数据预处理，验证集
            learningRate,  # learning rate
            shuffleFlag, numWorkers,
            modelYaml,  # yaml文件的名字, 如yolov8_1D-cls.yaml。位于conf/yolov8_1D目录, 其实放在conf目录就可以，会在conf文件夹根据文件名搜索.yaml文件
            scale,  # n s
            model,  # 模型参数文件（xxx_params.pt）的路径
            lossName,  # 损失函数
            optimizer  # 优化器，优化算法，用来更新模型参数
    ) -> None:
        super().__init__(
            dataset,
            epochNum,
            batchSize, sampleLength, step,
            transform,
            transform2,
            learningRate,  # learning rate
            shuffleFlag, numWorkers,
            modelYaml,
            scale,
            model,
            lossName,
            optimizer
        )
