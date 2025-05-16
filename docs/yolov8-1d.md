# YOLOv8_1D

由于采集的管道压力是一维数据。为了保证实时性并实现模型维度与数据维度的匹配，将YOLOv8分类网络`一维化`并调整卷积核尺寸，得到YOLOv8_1D模型。

和YOLOv8一样，YOLOv8_1D也有不同的 尺度/规模（scales）。

为了进一步提升模型性能，在`数据预处理`阶段，引入 经验小波变换（Empirical Wavelet Transform，EWT），更有效提取数据特征。
