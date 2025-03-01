# -*- coding: utf-8 -*-
# @Time    : 2024/3/29 17:07
# @author    : XiaoMo
# @Software: PyCharm
# 大鹏一日同风起，扶摇直上九万里

"""
torch.onnx.export(model, args, path, export_params, verbose, input_names, output_names, do_constant_folding, dynamic_axes, opset_version)
model——需要导出的pytorch模型
args——模型的输入参数，满足输入层的shape正确即可。
path——输出的onnx模型的位置。例如‘yolov5.onnx’。
export_params——输出模型是否可训练。default=True，表示导出trained model，否则untrained。
verbose——是否打印模型转换信息。default=False。
input_names——输入节点名称。default=None。
output_names——输出节点名称。default=None。
do_constant_folding——是否使用常量折叠，默认即可。default=True。
dynamic_axes——模型的输入输出有时是可变的，如Rnn，或者输出图像的batch可变，可通过该参数设置。如输入层的shape为（b，3，h，w），batch，height，width是可变的，但是chancel是固定三通道。
格式如下 ：
1)仅list(int) dynamic_axes={‘input’:[0,2,3],‘output’:[0,1]}
2)仅dict<int, string> dynamic_axes={‘input’:{0:‘batch’,2:‘height’,3:‘width’},‘output’:{0:‘batch’,1:‘c’}}
3)mixed dynamic_axes={‘input’:{0:‘batch’,2:‘height’,3:‘width’},‘output’:[0,1]}
opset_version——opset的版本，低版本不支持upsample等操作。
"""

import io
import torch
import torch.onnx
from cvit import CViT
import os

print(os.getcwd())


def test():
    model = CViT()

    pthfile = r'../weight/deepfake_cvit_gpu_inference_ep_50.pth'
    loaded_model = torch.load(pthfile, map_location='cpu')
    # try:
    #   loaded_model.eval()
    # except AttributeError as error:
    #   print(error)

    model.load_state_dict(loaded_model)

    # data type nchw
    input = torch.randn(1, 3, 224, 224)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, input, "cvit_model1.onnx", verbose=False, opset_version=12, input_names=input_names,
                      output_names=output_names)


if __name__ == "__main__":
    test()