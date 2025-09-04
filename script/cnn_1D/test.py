# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import sys
sys.path.append('.')
import os
from jyu.nn import MODEL_YAML_DEFAULT
import jyu.transform.transform as tf
from jyu.model.supervised.tester import Tester


def parse_args():
    parser = ArgumentParser()

    # netPath = os.path.join(os.getcwd(), 'result', 'train', '20240317.163818_Yolov8_1D', 'weights', 'best_params.pt')

    netPath = os.path.join('result', 'train', '20250302.120903_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250302.143001_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250302.163810_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250326.133420_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250603.175044_YI-Netv1', 'weights', 'best_params.pt')


    weight = netPath
    parser.add_argument('-w', '--weights', type=str, default=weight)

    parser.add_argument('-d', '--dataset', type=str, choices=['v4_press4', 'v4_wms', 'v1_wms'], default='v4_press4', help="数据集的名字，和yaml配置文件名相同，不包括.yaml后缀")
    parser.add_argument('-b', '--batchSize', type=int, default=64, help='Number of batch size to test.')
    parser.add_argument('-sl', '--sampleLength', type=int, default=4096, help='Data Length')
    parser.add_argument('-s', '--step', type=int, default=2048, help='Step Length')
    parser.add_argument('-t', '--transform', type=str,
                        choices=tf.testTransformList,
                        default=None, help='Transform for test sample')
    parser.add_argument('-sf', '--shuffleFlag', type=int, default=1, help='1 is True, 0 is False')
    # parser.add_argument('-n', '--numWorkers', type=int, default=0)
    parser.add_argument('-n', '--numWorkers', type=int, default=4)
    parser.add_argument('-my', '--modelYaml', type=str, default=None, help="模型对应的yaml文件")

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_args()
    tester = Tester(**vars(opt))
    tester.test()

if __name__ == '__main__':
    main()
