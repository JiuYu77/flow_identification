# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd())
from model.yolov8_1D.yolov8_1D import MODEL_YAML_S, MODEL_YAML_DEFAULT, yolov8_1d
# from model.yolov9_1D.yolov9_1D import yolov9_1d

__all__ = ["MODEL_YAML_S", "MODEL_YAML_DEFAULT", "yolov8_1d", "yolov9_1d"]
