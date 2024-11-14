# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd())
from model.yolo1d import YOLO1Dv8, YOLO1Dv10


__all__ = ["YOLO1Dv8", "YOLO1Dv10"]
