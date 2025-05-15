# -*- coding: UTF-8 -*-

import torch
import sys
sys.path.append(".")
from jyu.model import YI

weight = "/root/my_flow/flow_identification/result/train/20250326.133420_YI-Netv2/weights/best_params.pt"
# weight = "/root/my_flow/flow_identification/result/train/20250425.141300_YOLOv8_1D/weights/best_params.pt"
weight = "/root/my_flow/flow_identification/result/train/20250425.204828_YI-Netv2/weights/best_params.pt"
weight = "/root/my_flow/flow_identification/result/train/20250515.204010_YI-Netv2/weights/best_params.pt"
device = "cuda:0"

model = YI(weights=weight, device=device)
model.eval()

sample_shape = {"batch_size": 12, "channels": 1, "length": 4096}
dummy = torch.Tensor(sample_shape['batch_size'], sample_shape['channels'], sample_shape['length']).to(device)
dummy = torch.randn( sample_shape['batch_size'], sample_shape['channels'], sample_shape['length']).to(device)
print(dummy.shape, type(dummy), "dummy.size(2):", dummy.size(2))


root = "./result/ONNX"
net_name = model.netName
out_onnx = f"{root}/{net_name}-dynamic_axes.onnx"
input_names=["input"]
output_names=["output"]

dynamic = True  # True False
# dynamic = False  # True False

dynamic_axes={
    # "input": {0: "batch"},  # 第0维（batch）动态
    "input": {0: "batch", 2: "length"},  # 第0维（batch）动态
    "output": {0: "batch"}  # 输出同样支持动态batch
}
if dynamic is False:
    out_onnx = f"{root}/{net_name}-non_dynamic_axes.onnx"
    dynamic_axes = None

from jyu.utils import ph
ph.checkAndInitPath(root)

torch.onnx.export(
    model,
    (dummy,),
    f=out_onnx,
    input_names=input_names,
    output_names=output_names,
    opset_version=15,
    dynamic_axes=dynamic_axes,
)
