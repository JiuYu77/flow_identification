# -*- coding: UTF-8 -*-

import onnx

onnx_path = "script/ONNX/YI-Netv2-dynamic_axes.onnx"
model = onnx.load(onnx_path)

print("Input Nodes:")
for input in model.graph.input:
    print(input.name)
print("Output Nodes:")
for output in model.graph.output:
    print(output.name)
print("Nodes:")
for node in model.graph.node:
    print(node.name)

print("============================")

inferred_model = onnx.shape_inference.infer_shapes(model)

onnx.checker.check_model(inferred_model)

print(inferred_model.graph.value_info)
with open("script/ONNX/a.txt", "w") as f:
    f.write(str(inferred_model.graph.value_info))
