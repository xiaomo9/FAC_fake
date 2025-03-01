# -*- coding: utf-8 -*-
# @Time    : 2024/10/21 13:51
# @author    : XiaoMo
# @Software: PyCharm
# 大鹏一日同风起，扶摇直上九万里
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from cvit_GGCA_ADD_DEConv_RepBn8 import CViT

def main():
    model = CViT(image_size=224, patch_size=7, num_classes=2, channels=512,
                 dim=1024, depth=6, heads=8, mlp_dim=2048)
    checkpoint = torch.load(r'E:\Daima\face_fake\CViT-main\weight\4090RepBn8_dfdc_ff_total.pth', map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)
    target_layers = [model.features2[-3]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = r"E:\Daima\dataset\FF++\FF++\ff_face_total_data\train_data\Deepfakes\train\fake\002_006_3.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 0  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    model = CViT(image_size=224, patch_size=7, num_classes=2, channels=512,
                 dim=1024, depth=6, heads=8, mlp_dim=2048)
    # print(model)
    main()