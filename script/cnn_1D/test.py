# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import sys
sys.path.append('.')
import os
from jyu.nn import MODEL_YAML_DEFAULT
from jyu.model.supervised.tester import Tester


def parse_args():
    parser = ArgumentParser()

    # netPath = os.path.join(os.getcwd(), 'result', 'train', '20240317.163818_Yolov8_1D', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20241223.132744_YI-Net', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250208.175457_YOLOv8_1D', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250211.130638_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250211.153251_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250211.164607_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250211.172654_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250212.131921_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250212.143239_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250212.153916_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250212.185257_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250212.204855_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250212.204855_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250212.220312_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250212.220312_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250213.120924_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250213.120924_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250213.132627_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250213.132627_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250213.144954_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250213.144954_YI-Netv2', 'weights', 'last_params.pt')

    netPath = os.path.join('result', 'train', '20250213.161100_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250213.161100_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250213.172121_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250213.172121_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250214.091225_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250214.114823_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250214.114823_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250214.130849_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250214.130849_YI-Netv2', 'weights', 'last_params.pt')

    netPath = os.path.join('result', 'train', '20250214.202013_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250214.202013_YI-Netv2', 'weights', 'last_params.pt')

    netPath = os.path.join('result', 'train', '20250215.113313_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250215.113313_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250215.123842_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250215.123842_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250215.134301_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250215.134301_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250215.145254_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250215.161726_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250215.172719_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250215.172719_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250217.122102_YI-Netv1-PSA', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250217.140352_YI-Netv2-PSA', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250217.155846_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250217.155846_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250225.133024_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250225.154450_YI-Netv2', 'weights', 'last_params.pt')

    netPath = os.path.join('result', 'train', '20250225.191859_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250226.081708_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250226.081708_YI-Netv2', 'weights', 'last_params.pt')

    netPath = os.path.join('result', 'train', '20250226.105252_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250226.105252_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250226.125630_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250226.143830_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250226.143830_YI-Netv2', 'weights', 'last_params.pt')

    # netPath = os.path.join('result', 'train', '20250226.182253_YI-Netv2', 'weights', 'best_params.pt')
    # netPath = os.path.join('result', 'train', '20250226.182253_YI-Netv2', 'weights', 'last_params.pt')

    netPath = os.path.join('result', 'train', '20250227.103605_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250227.174029_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250227.214834_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250228.150814_YI-Netv2', 'weights', 'best_params.pt')
    netPath = os.path.join('result', 'train', '20250228.150814_YI-Netv2', 'weights', 'last_params.pt')
    netPath = os.path.join('result', 'train', '20250228.183336_YI-Netv2', 'weights', 'best_params.pt')


    weight = netPath
    parser.add_argument('-w', '--weights', type=str, default=weight)

    parser.add_argument('-d', '--dataset', type=str, choices=['v4_press4', 'v4_wms', 'v1_wms'], default='v4_press4', help="数据集的名字，和yaml配置文件名相同，不包括.yaml后缀")
    parser.add_argument('-b', '--batchSize', type=int, default=64, help='Number of batch size to test.')
    parser.add_argument('-sl', '--sampleLength', type=int, default=4096, help='Data Length')
    parser.add_argument('-s', '--step', type=int, default=2048, help='Step Length')
    parser.add_argument('-t', '--transform', type=str,
                        choices=['zScore_std',
                                 'normalization_MinMax',
                                 'multiple_zScore',
                                 'multiple_MinMax',
                                 'ewt_zScore',
                                 'dwt_zScore',
                                 'dwtg_zScore',
                                 'fft_zScore',
                                ],
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
