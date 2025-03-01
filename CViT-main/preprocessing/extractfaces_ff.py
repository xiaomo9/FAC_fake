# -*- coding: utf-8 -*-
# @Time    : 2024/6/3 13:57
# @author    : XiaoMo
# @Software: PyCharm
# 大鹏一日同风起，扶摇直上九万里
import os
import json
import numpy as np
import time
import face_recognition
import cv2
import shutil
from time import perf_counter
import sys
sys.path.insert(1, 'helpers')
sys.path.insert(1, 'model')
sys.path.insert(1, 'weight')
import helpers_face_extract_1
import helpers_read_video_1
import blazeface
import torch
from tqdm import tqdm
from blazeface import BlazeFace


# dir_path 是你的实际视频数据文件夹前一个文件夹的路径，因为这个文件夹下有多个文件夹，每个文件夹里是视频文件夹，他会遍历所有视频文件夹提取人脸。
dir_path = r"E:\mask\dataset\FF++\manipulated_sequences\DeepFakeDetection" # 待处理视频文件夹位置
image_path = r"E:\mask\dataset\FF++\ff_face_data\DeepFakeDetection" # 保存已提取人脸图片文件夹位置
print(os.getcwd())

def extract_face(dir_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    filenames = os.listdir(dir_path)
    for filename in tqdm(filenames,desc="Processing directories"):
        try:
            # 视频处理逻辑
            process_video(dir_path + '/' + filename, filename, image_path)
        except Exception as e:
            print(f"处理视频时发生错误: {dir_path + '/' + filename}, 错误信息: {str(e)}")

# access video
def process_video(video_path, filename, image_path): # 视频文件夹路径、视频具体名称、图片文件夹保存路径、
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    facedet = BlazeFace().to(gpu)
    facedet.load_weights("../helpers/blazeface.pth")
    facedet.load_anchors("../helpers/anchors.npy")
    _ = facedet.train(False)

    from helpers_read_video_1 import VideoReader
    from helpers_face_extract_1 import FaceExtractor
    # 每个视频取多少帧图像
    frames_per_video = 10
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_random_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn, facedet)

    faces = face_extractor.process_video(video_path)
    # 每帧只看一张脸。
    face_extractor.keep_only_best_face(faces)
    n = 0
    for frame_data in faces:
        for face in frame_data["faces"]:
            face_locations = face_recognition.face_locations(face)
            for face_location in face_locations:
                # 打印此图像中每个人脸的位置
                top, right, bottom, left = face_location
                # print（“人脸位于像素位置 顶部：{}，左：{}，底部：{}，右：{}”.format（顶部，左侧，底部，右侧））

                # 您可以像这样访问实际人脸本身：\n
                face_image = face[top:bottom, left:right]
                resized_face = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
                resized_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)

                cv2.imwrite(image_path + "/" + filename[:-4] + "_" + str(n) + ".jpg", resized_face,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 75])

                n += 1


start_time = perf_counter()
extract_face(dir_path)
end_time = perf_counter()
times = end_time - start_time
print("-------- 运行时间：{:.0f} m {:.0f} s --------".format(times // 60, times % 60))
