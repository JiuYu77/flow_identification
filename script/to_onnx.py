# -*- coding: UTF-8 -*-

import torch
import sys
sys.path.append(".")
from jyu.model import YI

weight = "/root/my_flow/flow_identification/result/train/20250326.133420_YI-Netv2/weights/best_params.pt"
device = "cuda:0"

model = YI(weights=weight, device=device)
model.eval()

batch_size = 12
sample_shape = [1, 4096]
dummy = torch.Tensor(batch_size, sample_shape[0], sample_shape[1]).to(device)
dummy = torch.randn(batch_size, sample_shape[0], sample_shape[1]).to(device)
print(dummy.shape, type(dummy), "dummy.size(2):", dummy.size(2))


torch.onnx.export(
    model,
    (dummy,),
    f="./script/cnn_1D/best_params.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=15,
    dynamic_axes={
        # "input": {0: "batch"},  # 第0维（batch）动态
        "input": {0: "batch", 2: "length"},  # 第0维（batch）动态
        "output": {0: "batch"}  # 输出同样支持动态batch
    },
)
