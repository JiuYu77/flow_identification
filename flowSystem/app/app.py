from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

import sys
sys.path.append('.')
from jyu.rknn.rknnlite import RknnLite
from jyu.dataset.flowDataset import FlowDataset
from jyu.dataloader.flowDataLoader import FlowDataLoader

app = Flask(__name__)
CORS(app) # 跨域问题 (CORS), 前端页面和后端服务不在同一个域名下，可能会遇到 CORS 问题

# 假设有一个深度学习模型对象 model
# from model import model

# 流型名称列表
FLOW_TYPES = [
    "段塞流",
    "伪段塞流",
    "分层波浪流",
    "分层光滑流",
    "泡沫段塞流",
    "分层泡沫波浪流",
    "泡沫环状流"
]
rknn_model = "result/rknn/20250603.175044_YI-Netv1/YI-Netv1-dynamic_axes.rknn"
rknn_lite = RknnLite(rknn_model)

datasetPath = "/home/orangepi/flow/dataset/flow/v4/Pressure/4/val"
transformName = "zScore_std"
batchSize = 31
shuffle=True
sampleLength = 0
step = 0

dataset = None
dataloader = None

@app.route('/', methods=['GET'])
def index():
    return "Hello, World!"

@app.route('/api/predict', methods=['POST'])
def predict():
    # 声明全局变量
    global sampleLength, step, dataset, dataloader

    req = request.json
    length = int(req['sample_length'])
    stp = int(req['step'])
    if sampleLength != length or step != stp:
        sampleLength = length
        step = stp
        dataset = FlowDataset(datasetPath, sampleLength, step, transformName, clss=None, supervised=True, toTensor=False)
        dataloader = FlowDataLoader(dataset, batchSize, shuffle)

    print("sampleLength=", sampleLength, "step=", step)

    results = []
    pred_label = 0
    ii = 0

    for X, Y in dataloader:
        if ii > 0:
            break
        ii += 1
        X = X.astype(np.float32)
        outputs = rknn_lite(X) # 推理
        # 后处理（示例：分类模型）
        pred_label = np.argmax(outputs, axis=2)
        y_hat = np.array(outputs)
        print("Predicted class:", pred_label)
        print("Predicted class:", pred_label[0])
        print("Predicted class:", int(pred_label[0][0]))

    print("Predicted class out:", pred_label)
    print("Predicted class out:", pred_label[0])
    print("Predicted class out:", int(pred_label[0][0]))

    preLabel = int(pred_label[0][0])  # 预测标签
    x = X[0][0].tolist()
    results.append({
        "preLabel": preLabel,
        "flowType": FLOW_TYPES[preLabel],
        "flowData": x,
    })

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
