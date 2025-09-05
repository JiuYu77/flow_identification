# -*- coding: UTF-8 -*-
from rknnlite.api import RKNNLite


class RknnLite:
    def __init__(self, rknn_model, core_mask=RKNNLite.NPU_CORE_AUTO):
        self.rknn_lite = RKNNLite()  # 初始化RKNN Lite

        r = self.rknn_lite.load_rknn(rknn_model)  # 加载模型
        assert r == 0, "Load ONNX failed!"

        self.core_mask = core_mask
        self._init_runtime()  # 初始化运行时
    
    def _init_runtime(self):
        '''初始化运行时'''
        ret = self.rknn_lite.init_runtime(core_mask=self.core_mask)  # 指定NPU核心
        assert ret == 0, "Init runtime failed!"

    def __call__(self, inputs, *args, **kwds):
        outputs = self.rknn_lite.inference(inputs=[inputs])  # 执行推理
        return outputs

    def release(self):
        self.rknn_lite.release()
