# -*- coding: utf-8 -*-
"""散点图"""
import sys
sys.path.append('.')
from jyu.utils import plot
from jyu.utils.plot.getdata import ScatterData


path = [
        # "/home/uu/my_flow/flow_identification/result/train/20240311.190336_Yolov8_1D/",
        # "/home/uu/my_flow/flow_identification/result/train/20240314.155221_Yolov8_1D/",
        "/home/uu/my_flow/flow_identification/result/train/20240317.163818_Yolov8_1D/",
        # "/home/uu/my_flow/flow_identification/",
    ]

if __name__ == '__main__':
    if type(path) == list:
        for p in path:
            scatter = plot.Scatter(p, s=10)
    else:
        scatter = plot.Scatter(path, s=10, figsize=(8, 7))
    scatter.draw_save()

