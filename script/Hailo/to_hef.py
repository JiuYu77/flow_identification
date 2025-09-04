# -*- coding: UTF-8 -*-
"""
将 onnx 模型 转换未 hef 格式
  1. onnx模型 转换成 Hailo Archive 模型（.har）。
  2. 将 .har 模型进行量化。
  3. 编译为 Hailo Executable File 模型（.hef）。
"""

from hailo_sdk_client import ClientRunner

import hailo_sdk_common.hailo_nn
import hailo_sdk_common.hailo_nn.hailo_nn
import hailo_sdk_common.hailo_nn.hn_layers
import hailo_model_optimization, hailo_sdk_client, hailo_sdk_server, hailo_sdk_common, hailo_tools, hailo_tutorials

hailo_sdk_common.hailo_nn.hailo_nn.Conv2DLayer

def to_har(
        runner:ClientRunner, onnx_path, onnx_model_name, out_har_path,
        start_node_names=None,
        end_node_names=None,
        net_input_shapes=None
    ) ->tuple:
    '''
    .onnx 转为 .har
    Args:
        hw_arch(str, optional): 要使用的硬件架构，默认为“hailo8”。

    Returns:
        tuple: The first item is the HN JSON as a string. The second item is the params dict.  
        第一项是字符串形式的 HN JSON。第二项是参数字典。
    '''
    hn_data, native_npz = runner.translate_onnx_model(
        model=onnx_path,
        net_name=onnx_model_name,
        start_node_names=start_node_names,
        end_node_names=end_node_names,
        net_input_shapes=net_input_shapes
    )
    runner.save_har(out_har_path)
    return hn_data, native_npz


def do_quantization(runner:ClientRunner, calib_dataset, hailo_quantized_har_path):
    '''量化'''
    alls_lines = [
        'model_optimization_flavor(optimization_level=1, compression_level=2)',
        'resources_param(max_control_utilization=0.6, max_compute_utilization=0.6, max_memory_utilization=0.6)',
        'performance_param(fps=5)'
    ]
    runner.load_model_script('\n'.join(alls_lines))
    runner.optimize(calib_dataset)
    runner.save_har(hailo_quantized_har_path)

def do_compile(runner:ClientRunner, hailo_model_hef_path):
    '''编译为 hef'''
    compiled_hef = runner.compile()
    with open(hailo_model_hef_path, "wb") as f:
        f.write(compiled_hef)

def to_hef():
    '''转为.hef'''
    hw_arch = "hailo8" # None -> 'hailo8'
    root = "./script/Hailo"
    onnx_path = "script/ONNX/YI-Netv2-dynamic_axes.onnx"
    # onnx_path = "script/ONNX/YI-Netv2-non_dynamic_axes.onnx"

    onnx_model_name = "YI-Netv2"
    start_node_names = ["input"]
    # input_shape = [1, 1, 4096]
    input_shape = { start_node_names[0]: [1, 1, 4096] }
    input_shape = { start_node_names[0]: [12, 1, 4096] }
    end_node_names = None
    # end_node_names = ["Add_43", "Slice_22", "Transpose_103", "MatMul_107", "Slice_25"]
    # end_node_names = ["Slice_32", "Slice_29", "MatMul_114", "Transpose_110", "Add_50"]
    end_node_names = ["MatMul_71", "Add_50", "Slice_32", "Add_68", "MatMul_69", "Slice_29"]
    # end_node_names = [ "Add_44", "Slice_29", "Add_53", "Slice_32", "Add_49", "Add_51"]
    # end_node_names = ["Slice_29", "Add_44", "Add_53", "Add_49", "Add_51", "Slice_32"]
    # end_node_names = [ "Slice_22", "Conv_3", "Add_65", "Add_63", "Add_61", "Slice_25", "Add_43"]
    # end_node_names = ["Slice_22", "Conv_3", "Add_61", "Add_63", "Add_43", "Slice_25", "Add_65"]
    # end_node_names = ["Slice_32", "Add_44", "Slice_29", "Add_53", "Add_51", "Add_49"]
    hailo_model_har_path = f"{root}/{onnx_model_name}_hailo_model.har"  # 转换后模型的保存路径
    hailo_quantized_har_path = f"{root}/{onnx_model_name}_hailo_quantized_model.har"  # 量化后模型的保存路径
    hailo_model_hef_path = f"{root}/{onnx_model_name}.hef"  # 编译后模型的保存路径

    # 1. 将 onnx 模型转为 har
    runner = ClientRunner(hw_arch=hw_arch)
    to_har(
        runner, onnx_path,
        onnx_model_name=onnx_model_name,
        out_har_path=hailo_model_har_path,
        start_node_names=start_node_names,
        end_node_names=end_node_names,
        net_input_shapes=input_shape
    )

    # 2. 量化模型
    # runner = ClientRunner(har=hailo_model_har_path)
    # do_quantization(runner, hailo_quantized_har_path)

    # 3. 编译为 hef
    runner = ClientRunner(har=hailo_quantized_har_path)
    do_compile(runner, hailo_model_hef_path)

to_hef()
