# YI-Net

由于 经验小波变换（Empirical Wavelet Transform，EWT）时间开销过大，所以在数据预处理阶段去除EWT，可进一步提升模型的`实时性`。

为了保证模型性能，对模型做出改进：提出 `C2fTR1d` 模块，该模块继承字 C2f 模块，并将 `nn.ModuleList`（用于添加 Bottleneck模块）模块，修改为一个`TransformerBlock`模块，以保证 识别/分类 准确率。

这是YI-Net网络的第2个版本，记为**YI-Netv2**。

