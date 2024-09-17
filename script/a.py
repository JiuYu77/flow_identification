# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
from model import *
from utils.plot import plot
import matplotlib.pyplot as plt
import numpy as np
import torch
import random


net = YOLOv8_1D('yolov8_1Ds-cls.yaml', fuse_=True)
print(net.scale, type(net.scale))
print(net.__class__.__name__)

print(torch.__version__)

