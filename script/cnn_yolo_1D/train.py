# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
from argparse import ArgumentParser
from model import MODEL_YAML_DEFAULT
from utils import print_color
from model.yolo.trainer import BaseTrainer


def parse_args():
    print_color(["black", "bold", "process args..."])
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['v4_press4', 'v4_press3', 'v4_wms', 'v1_wms'], default='v4_press4', help='dataset name')
    parser.add_argument('-e', '--epochNum', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('-b', '--batchSize', type=int, default=64, help='Number of batch size to train.')
    parser.add_argument('-sl', '--sampleLength', type=int, default=4096, help='Data Length')
    parser.add_argument('-st', '--step', type=int, default=2048, help='Step Length')
    parser.add_argument('-t', '--transform', type=str,
                        choices=['standardization_zScore',
                                 'normalization_MinMax',
                                 'multiple_zScore',
                                 'multiple_MinMax',
                                 'std_gaussianNoise',
                                 'ewt_std',
                                 'dwt_std',
                                 'dwtg_std',
                                 'fft_std',
                                ],
                        default="standardization_zScore", help='Transform for train sample')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('-sf', '--shuffleFlag', type=int, default=1, help='1 is True, 0 is False')
    parser.add_argument('-n', '--numWorkers', type=int, default=4)
    parser.add_argument('-my', '--modelYaml', type=str, default=MODEL_YAML_DEFAULT, help='yaml文件名, 如yolov8_1D-cls.yaml')
    parser.add_argument('-sc', '--scale', type=str, default=None)
    parser.add_argument('-m', '--model', type=str, default=None, help="模型参数文件的路径, best_params.pt")
    parser.add_argument('-ls', '--lossName', type=str, default='CrossEntropyLoss', help="损失函数")
    parser.add_argument('-op', '--optimName', type=str, choices=['SGD', 'Adam', 'AdamW', 'LION'] , default='SGD', help="优化器，优化算法，用来更新模型参数")
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_args()
    trainer = BaseTrainer(**vars(opt))
    trainer.train()


if __name__ == '__main__':
    main()
