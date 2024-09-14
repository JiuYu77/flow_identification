# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import sys
sys.path.append('.')
from model import MODEL_YAML_DEFAULT
import os
from model.yolo.tester import BaseTester


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

def main():
    opt = parse_args()
    tester = BaseTester(**vars(opt))
    tester.test()
    
if __name__ == '__main__':
    main()
