# -*- coding: utf-8 -*-
# @Time    : 2024/6/4 14:52
# @author    : XiaoMo
# @Software: PyCharm
# 大鹏一日同风起，扶摇直上九万里
import os
import random
from PIL import Image

def split_images(input_folder, output_folder1, output_folder2):
    # 获取文件夹中所有图片文件的路径
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)

    # 确定每个输出文件夹中应该包含的图片数量
    num_images1 = int(len(image_files) * 0.85)  # 4/5
    num_images2 = len(image_files) - num_images1  # 1/5

    # 创建输出文件夹
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)


    # 分配图片到两个输出文件夹
    for i, image_file in enumerate(image_files):
        input_path = os.path.join(input_folder, image_file)
        if i < num_images1:
            output_path = os.path.join(output_folder1, image_file)
        else:
            output_path = os.path.join(output_folder2, image_file)

        # 检查目标文件夹中是否存在同名文件，如果存在，则更改文件名
        if os.path.exists(output_path):
            base_name, ext = os.path.splitext(image_file)
            index = 1
            while os.path.exists(os.path.join(output_folder1, f"{base_name}_{index}{ext}")):
                index += 1
            output_path = os.path.join(output_folder1, f"{base_name}_{index}{ext}")

        # 复制图片到对应的输出文件夹
        with Image.open(input_path) as img:
            img.save(output_path)

    print("Images have been split successfully!")

# 用法示例
input_folder = r"E:\Daima\dataset\Celeb-DF-v2\total_face\YouTube-real"
output_folder1 = r"E:\Daima\dataset\Celeb-DF-v2\total_face\total\train\real"
output_folder2 = r"E:\Daima\dataset\Celeb-DF-v2\total_face\total\validation\real"

split_images(input_folder, output_folder1, output_folder2)