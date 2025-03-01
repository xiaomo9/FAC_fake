# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 10:54
# @author    : XiaoMo
# @Software: PyCharm
# 大鹏一日同风起，扶摇直上九万里
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
with open(r'E:\Daima\face_fake\CViT-main\weight\cvit_GGCA_ADD_DEConv_RepBn8_FF_NeuralTextures.pkl', 'rb') as file:
    data = pickle.load(file)
lens = len(data[1])
datas1 = data[1]
datas2 = data[3]
print(max(datas1))
print(max(datas2))
# 查看文件内容
# print(data)
plt.plot(range(lens), datas1, label='train_loss')
plt.plot(range(lens), datas2, label='train_loss')
plt.show()

#===========================================================================================================
