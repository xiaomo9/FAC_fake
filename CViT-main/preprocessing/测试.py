import sys
import os
import json
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
sys.path.insert(1,'helpers')
sys.path.insert(1,'model')
sys.path.insert(1,'weight')

import pickle
# matplotlib.use('TkAgg')
# ==========================================读取.pkl文件====================================================
with open('cvit_deepfake_detection_111.pkl', 'rb') as file:
    data = pickle.load(file)
lens = len(data[1])
datas = data[3]
print(max(datas))
# 查看文件内容
# print(data)
plt.plot(range(lens), datas, label='train_loss')
plt.show()

#===========================================================================================================
