# -*- coding: UTF-8 -*-
from rknn.api import RKNN


class ToRKNN:
    def __init__(self, rknnArgs:dict, configArgs:dict, loadArgs:dict, buildArgs:dict, exportArgs:dict):
        self.rknn = RKNN(**rknnArgs)  # 初始化转换器
        self.rknn.config(**configArgs)  # 配置模型参数
        self.loadArgs = loadArgs
        self.buildArgs = buildArgs
        self.exportArgs = exportArgs

    def _load_onnx(self): # 加载ONNX模型
        r = self.rknn.load_onnx(**self.loadArgs)
        assert r == 0, "Load ONNX failed!"

    def _build(self): # 转换模型
        # ret = rknn.build(do_quantization=True, dataset="dataset.txt")  # 这一行是备注，量化需提供校准数据集
        r = self.rknn.build(**self.buildArgs) 
        assert r == 0, "Build RKNN failed!"

    def _export_rknn(self): # 导出RKNN模型
        r = self.rknn.export_rknn(**self.exportArgs)
        assert r == 0, "Export RKNN failed!"    

    def release(self): # 释放资源
        self.rknn.release()

    def run(self):
        self._load_onnx()
        self._build()
        self._export_rknn()

