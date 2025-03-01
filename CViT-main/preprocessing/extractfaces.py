import os
os.chdir('/home/AIBike_ViG/interns/chenao/rknnkit')
import json
import numpy as np
import time
import face_recognition
import cv2
import shutil
from time import perf_counter
import sys
sys.path.insert(1, '/home/AIBike_ViG/interns/chenao/rknnkit/test/CViT-main/helpers')
sys.path.insert(1, '/home/AIBike_ViG/interns/chenao/rknnkit/test/CViT-main/model')
sys.path.insert(1, '/home/AIBike_ViG/interns/chenao/rknnkit/test/CViT-main/weight')
import helpers_face_extract_1
import helpers_read_video_1
import blazeface
import torch
from tqdm import tqdm
from blazeface import BlazeFace

print(os.getcwd())
# dir_path 是你的实际视频数据文件夹前一个文件夹的路径，因为这个文件夹下有多个文件夹，每个文件夹里是视频文件夹，他会遍历所有视频文件夹提取人脸。
dir_path = "/home/AIBike_ViG/interns/chenao/rknnkit/data/dfdc_train_part_014/"
train_path = "dfdc_data/training_face_"
validation_path = "dfdc_data/validation_face_"
test_path = "dfdc_data/test_face_"


# load DFDC json
def load_metadata(dir_path):
    metafile = dir_path + 'metadata.json'

    if os.path.isfile(metafile):
        with open(metafile) as data_file:
            data = json.load(data_file)
    else:
        return 1

    return data


def extract_face(dir_path):
    # iterate over DFDC dataset
    for item in sorted(os.listdir(dir_path)):

        if not item[16:]:
            continue
        file_num = int(item[16:])
        destination = train_path

        if (file_num > 34 and file_num < 45):
            destination = test_path

        if (file_num > 45):
            destination = validation_path

        #  dfdc_data/dfdf_train_part_3
        meta_full_path = os.path.join(dir_path, item)

        if os.path.isdir(meta_full_path):
            # dfdc_data/dfdf_train_part_3/
            data = load_metadata(meta_full_path + '/')

            if data != 1:

                if not os.path.exists(destination + str(file_num)):
                    # dfdc_data/training_face_3 这个按照文件内容而变根据数字范围分为：train、validation、test
                    os.makedirs(destination + str(file_num))

                # 查看新建的training_face_3文件是否存在metadata.json,若不存在，则复制原始文件里的。json文件到本文件夹里。
                if not os.path.isfile(destination + str(file_num) + '/metadata.json'):
                    shutil.copy2(dir_path + item + '/metadata.json', destination + str(file_num) + '/metadata.json')

                # 该代码的功能是对给定的数据进行筛选，去除重复的文件
                # data=dfdc_data/dfdf_train_part_3/
                filtered_files = filter_unique_files(data)#一个原始视频只保留一个假视频

                # 若想要处理所有视频的话将下面的filtered_files换成data即可
                for filename in tqdm(filtered_files, desc="Processing directories"):
                    # 检查是否在元数据中找到文件名及其标签
                    if filename.endswith(".mp4") and os.path.isfile(dir_path + item + '/' + filename):
                        label = data[filename]['label'].lower()
                        original = ''
                        if data[filename]['label'].lower() == 'fake':
                            original = '_' + data[filename]['original'][:-4]
                        image_path = destination + str(file_num) + '/' + label
                        if not os.path.exists(image_path):
                            os.makedirs(image_path)

                        # 处理视频，提取每个视频中所含人脸
                        try:
                            # 视频处理逻辑
                            process_video(dir_path + item + '/' + filename, filename, image_path, original)
                        except Exception as e:
                            print(f"处理视频时发生错误: {dir_path + item + '/' + filename}, 错误信息: {str(e)}")



# access video
def process_video(video_path, filename, image_path, original):
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    facedet = BlazeFace().to(gpu)
    facedet.load_weights("/home/AIBike_ViG/interns/chenao/rknnkit/test/CViT-main/helpers/blazeface.pth")
    facedet.load_anchors("/home/AIBike_ViG/interns/chenao/rknnkit/test/CViT-main/helpers/anchors.npy")
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

                cv2.imwrite(image_path + "/" + filename[:-4] + original + "_" + str(n) + ".jpg", resized_face,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 75])

                n += 1


def filter_unique_files(metadata):#将由一个原始视频伪造的所有假视频只保留最前面的那个视频
    fake = []
    original = []

    for file_dp in metadata:
        if (('original' in metadata[file_dp]) and (metadata[file_dp]['original'] not in original) and (
                metadata[file_dp]['original'] is not None)):
            original.append(metadata[file_dp]['original'])
            fake.append(file_dp)
    return np.array([[i, j] for i, j in zip(fake, original)]).ravel()


start_time = perf_counter()
extract_face(dir_path)
end_time = perf_counter()
times = end_time - start_time
print("-------- 运行时间：{:.0f} m {:.0f} s --------".format(times // 60, times % 60))
