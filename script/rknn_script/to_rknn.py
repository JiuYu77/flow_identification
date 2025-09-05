# -*- coding: UTF-8 -*-
import sys
sys.path.append(".")
from jyu.utils import ph
from jyu.rknn.rknn import ToRKNN


dynamic = True
# dynamic = False

time_dir = "20250603.175044_YI-Netv1"
model_name = "YI-Netv1"

# 配置模型参数
if dynamic:
    dynamic_input=[[[1, 1, 4096]], [[12, 1, 4096]], [[31, 1, 4096]]]
    model_name += "-dynamic_axes"
else:
    dynamic_input=None
    model_name += "-non_dynamic_axes"

onnx_model = model_name + ".onnx"
model = f"result/ONNX/{time_dir}/{onnx_model}"

root = "./result/rknn/" + time_dir
export_path = f"{root}/{model_name}.rknn"
ph.checkAndInitPath(root)

###############
# from rknn.api import RKNN
# rknn = RKNN(verbose=True)
# rknn.config(
#     # mean_values=[[123.675, 116.28, 103.53]],  # ImageNet标准化参数
#     # std_values=[[58.395, 58.395, 58.395]],
#     mean_values=[[123.675]],  # ImageNet标准化参数
#     std_values=[[58.395]],
#     target_platform="rk3588",
#     quantized_dtype="asymmetric_quantized-8",  # 启用INT8量化
#     dynamic_input=dynamic_input
# )
###############


# 转换模型
trknn = ToRKNN(
    {"verbose":True},
    {
        "mean_values":[[123.675]],  # ImageNet标准化参数
        "std_values": [[58.395]],
        "target_platform":"rk3588",
        "quantized_dtype":"asymmetric_quantized-8",  # 启用INT8量化
        "dynamic_input":dynamic_input
    },
    {"model":model},
    {"do_quantization":False},
    {"export_path":export_path}
)

trknn.run()
trknn.release()
