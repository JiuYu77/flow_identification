# YI-Net

由于 经验小波变换（Empirical Wavelet Transform，EWT）时间开销过大，所以在数据预处理阶段去除EWT，可进一步提升模型的`实时性`。

为了减少YOLOv8_1D的`冗余计算`，将YOLOv10提出的CIB模块一维化为CIB1d，并用CIB1d取代C2f1d的Bottleneck层，得到`C2fCIB1d`模块，用此模块代替YOLOv8_1D配置文件的最后一个C2f1d。

为了保证模型性能，对模型做出改进：提出 `C2fTR1d` 模块，该模块继承字 C2f 模块，并将 `nn.ModuleList`（用于添加 Bottleneck模块）模块，修改为一个`TransformerBlock`模块，以保证 识别/分类 准确率。用此模型代替YOLOv8_1D配置文件中的前两个C2f1d。

为了实现进一步轻量化，减少模型`参数量`，将YOLOv10提出的SCDown模块一维化为`SCDown1d`，用此模块代替YOLOv8_1D配置文件中的后两个卷积Conv1d。

最终得到新型一维模型，命名为**YI-Net**，全称为 “**One-dimensional Intelligent Network**”，即“*一维智能网络*”。

这是YI-Net网络的第1个版本，记为**YI-Netv1**。

