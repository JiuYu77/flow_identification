# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
from model.yolo1d.yolov8_1D import MODEL_YAML_S, MODEL_YAML_DEFAULT, YOLOv8_1D
from model.yolo1d.yolov10_1D import YOLOv10_1D

__all__ = ['MODEL_YAML_S', 'MODEL_YAML_DEFAULT', 'YOLOv8_1D', 'YOLOv10_1D']
