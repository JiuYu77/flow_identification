# -*- coding: UTF-8 -*-
"""自组织映射 (Self-Organizing Map, SOM)"""

from minisom import MiniSom
import numpy as np


def do_som(data, x, y, input_len, learning_rate=0.5):
    print("do_som...")
    som = MiniSom(x=x, y=y, input_len=input_len, learning_rate=learning_rate)
    som.random_weights_init(data) # 随机初始化权重
    print("   som train...")
    som.train_random(data, num_iteration=100) # 训练

    # low_dim_data = som.distance_map() # 降维数据
    print("   som winner...")
    bmus = np.array([som.winner(x) for x in data])

    return bmus
