# YI-Net

为了进一步降低YOLOv8_1D的参数量，将YOLOv10提出的SCDown模块一维化为SCDown1d，用此模块代替YOLOv8_1D配置文件中的后两个卷积Conv1d。

为了减少YOLOv8_1D的冗余计算，将YOLOv10提出的CIB模块一维化为CIB1d，并用CIB1d取代C2f1d的Bottleneck层，得到C2fCIB1d模块，用此模块代替YOLOv8_1D配置文件的最后一个C2f1d。

最终得到新型一维模型，命名为**YI-Net**，全称为 “*One-dimensional Intelligent Network*”，即“*一维智能网络*”。

这是YI-Net网络的第一个版本，记为YI-Netv1。

# YI-Net-PSA

将YOLOv10的PSA模块一维化，并将此模块添加到YI-Netv1中，得到YI-Net-PSA。

与YI-Netv1相比，YI-Net-PSA提升了一定的特征提取能力，缺点是模型参数量增加，比YOLOv8_1D的参数量还要多。
