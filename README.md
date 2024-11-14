# encoding
UTF-8

# Description
这是一个基于深度学习的管道流型识别项目。参考了YOLO系列模型，YOLOv8、YOLOv10等。

**YOLO1Dv8**、**`YOLO1Dv10`**。

# clone
```bash
git clone https://github.com/JiuYu77/flow_identification.git
```

# 配置环境
（1） 安装PyTorch以外的库
```bash
pip install -r requirements.txt
```
（2） 安装PyTorch

# 数据集格式
提供了两种数据集结构：（1）训练集、验证集和测试集（推荐）；（2）训练集和测试集

## 结构1 训练集、验证集和测试集（推荐）
```text
  Pressure_Simple
  ├── train--------训练集
  │   ├── 0
  │   ├── 1
  │   ├── 2
  │   ├── 3
  │   ├── 4
  │   ├── 5
  │   └── 6
  ├── val----------验证集
  │   ├── 0
  │   ├── 1
  │   ├── 2
  │   ├── 3
  │   ├── 4
  │   ├── 5
  │   └── 6
  └── test---------测试集
      ├── 0
      ├── 1
      ├── 2
      ├── 3
      ├── 4
      ├── 5
      └── 6
```

## 结构2 训练集和测试集
测试集和验证集是数据集的同一部分，test=val。适用于数据不足的情况。
```text
  Pressure_Simple
  ├── train--------训练集
  │   ├── 0
  │   ├── 1
  │   ├── 2
  │   ├── 3
  │   ├── 4
  │   ├── 5
  │   └── 6
  └── val---------测试集
      ├── 0
      ├── 1
      ├── 2
      ├── 3
      ├── 4
      ├── 5
      └── 6
```

# bash脚本

具体使用方法：查看帮助信息。
```shell
bash cmd.sh help
```
