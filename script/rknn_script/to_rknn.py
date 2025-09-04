# -*- coding: UTF-8 -*-

from rknn.api import RKNN
import sys
sys.path.append(".")
from jyu.utils import ph


# 初始化转换器
rknn = RKNN(verbose=True)

dynamic = True
# 配置模型参数
if dynamic:
    dynamic_input=[[[1, 1, 4096]], [[12, 1, 4096]], [[31, 1, 4096]]]
else:
    dynamic_input=None

rknn.config(
    # mean_values=[[123.675, 116.28, 103.53]],  # ImageNet标准化参数
    # std_values=[[58.395, 58.395, 58.395]],
    mean_values=[[123.675]],  # ImageNet标准化参数
    std_values=[[58.395]],
    target_platform="rk3588",
    quantized_dtype="asymmetric_quantized-8",  # 启用INT8量化
    dynamic_input=dynamic_input
)

# 加载ONNX模型
time_dir = "20250603.175044_YI-Netv1"
model_name = "YI-Netv1"

if dynamic:
    model_name += "-dynamic_axes"
else:
    model_name += "-non_dynamic_axes"

onnx_model = model_name + ".onnx"
model = f"result/ONNX/{time_dir}/{onnx_model}"
ret = rknn.load_onnx(model=model)
if ret != 0:
    print("Load ONNX failed!")
    exit(ret)

# 转换模型
# ret = rknn.build(do_quantization=True, dataset="dataset.txt")  # 量化需提供校准数据集
# ret = rknn.build()
ret = rknn.build(do_quantization=False)
if ret != 0:
    print("Build RKNN failed!")
    exit(ret)

# 导出RKNN模型
root = "./result/rknn/" + time_dir
export_path = f"{root}/{model_name}.rknn"
ph.checkAndInitPath(root)

ret = rknn.export_rknn(export_path)
if ret != 0:
    print("Export RKNN failed!")
    exit(ret)

# 释放资源
rknn.release()

