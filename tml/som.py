# -*- coding: UTF-8 -*-
"""自组织映射 (Self-Organizing Map, SOM)"""

from minisom import MiniSom
import numpy as np

som = MiniSom(x=10, y=10, input_len=4096, learning_rate=0.5)

data = []
data = np.random.rand(2, 100)
som = MiniSom(x=2, y=10, input_len=100, learning_rate=0.5)

som.random_weights_init(data) # 随机初始化权重
som.train_random(data, num_iteration=100) # 训练

low_dim_data = som.distance_map() # 降维数据
print(low_dim_data)
print(type(low_dim_data))
with open("tml/bbb.txt", 'w') as f:
    for v in low_dim_data:
        ss = str(v)+'\n'
        f.write(ss)
