# -*- coding: utf-8 -*-
import os


ACC_IDX = 4
LOSS_IDX = 5
AVGLOSS_IDX = 6


class ScatterData:
    """读取训练结果数据"""
    def __init__(self, path:str) -> None:
        """
        path: 训练结果路径
        """
        assert os.path.isdir(path), f"{path}文件夹不存在"
        self.path = path

    @staticmethod
    def read_data(path, idx, head:bool=True):
        """
        从path指定的文件中读取第idx列数据数据,idx从0开始
        head: True表示文件第一行为标头，跳过标头
        """
        assert os.path.isfile(path), f"{path}文件不存在"
        data = []
        with open(path) as fp:
            lines = fp.readlines()
        for line in lines:
            if head:
                head = False
                continue
            line = line.strip('\n')
            item = line.split('\t')
            # print(idx+1, "  ", len(item))
            assert idx + 1 <= len(item), f"数据字段数量错误"
            data.append(float(item[idx]))
        return data

    def get_data(self, idx, train:bool=True):
        """
        train: True 返回训练结果相关数据
               False 返回验证结果相关数据
        """
        assert type(train) is bool, "train must be bool"
        if train:
            path = os.path.join(self.path, 'train_iter')
            data = self.read_data(path, idx=idx)
        else:
            path = os.path.join(self.path, 'val_iter')
            data = self.read_data(path, idx=idx)
        return data

    def get_avgloss(self, train:bool=True):
        """
        train: True 返回训练结果的每个batch的平均损失
               False 返回验证结果的每个batch的平均损失
        """
        avgLoss = self.get_data(AVGLOSS_IDX, train=train)
        return avgLoss

    def get_loss(self, train:bool=True):
        """
        train: True 返回训练结果的每个batch的总损失
               False 返回验证结果的每个batch的总损失
        """
        loss = self.get_data(LOSS_IDX, train=train)
        return loss
 
    def get_acc(self, train:bool=True):
        """
        train: True 返回训练结果的准确率
               False 返回验证结果的准确率
        """
        acc = self.get_data(ACC_IDX, train=train)
        return acc

    def get_xlim(self):
        pass
 
    def get_ylim(self):
        pass
