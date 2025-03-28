# encoding
UTF-8

# Paper
**[Lightweight Real-Time Network for Multiphase Flow Patterns Identification Based on Upward inclined Pipeline Pressure Data](https://www.sciencedirect.com/science/article/abs/pii/S0955598625000329)**

# Description
人工智能用于流型识别：一维模型（AI for Flow Pattern Identification: One-dimensional Model）

这是一个基于深度学习的**管道流型识别**项目，提供了多个一维分类模型。采用一维序列数据集。

本项目构建的一维网络架构：**`YOLOv8_1D`**、**`YI-Net`**。

此项目不仅可以用于流型识别分类任务，在样本数据点足够多（≥500）的条件下，通常也适用于其他一维分类工作。

参考了YOLO系列模型，YOLOv8、YOLOv10等。

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

# 数据集
本项目提出的YOLOv8_1D等模型，采用一维输入样本。

将一维数据集，按照某个数值*length*（如2048、4096等），划分为多个数据点数为*length*的一维样本。一个一维样本是一个行向量。

## 数据集格式
提供了两种数据集结构：（1）训练集、验证集和测试集（推荐）；（2）训练集和测试集

### 结构1 训练集、验证集和测试集（推荐）
0、1...6既是文件夹名，又是标签，称为标签文件夹。

每个标签文件夹下可以有多个文本文件，每个文本文件只有一列数据。

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

### 结构2 训练集和测试集
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

### 自定义数据集
你也可以使用，其他数据集格式，例如.txt、.tsv、.csv、.xls、.xlsx等文件存储的数据集。只要输入到模型中的样本是行向量（张量）就可以。

你需要编写自己的**Dataset**类，或者将处理好的样本与标签赋值给**FlowDataset**类的allSample、allLabel属性。

# bash脚本

具体使用方法：查看帮助信息。
```shell
bash cmd.sh help
```
