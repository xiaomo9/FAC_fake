import math
import random
from tqdm import tqdm
import cv2
import face_alignment
import numpy as np
import os
import time

# 输入图片，得到对应的掩码处理后的结果。
# 图片格式为cv2读取后的格式，shape为（H，W，C），颜色通道顺序为CV2的BGR（这个没什么关系）。
# mask_method='black' or 'noise'，前者为黑色填充，后者为高斯噪声填充。
def get_masked_face(input_img, face_align, mask_method, file_name):

    if mask_method not in ['black', 'noise']:
        print("please input mask_method('black' or 'noise').\nno change for input image.")
        return input_img
    
    #默认使用GPU，如果要在CPU上运行，则需要添加参数device='cpu'。
    #face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=dev)
    #获取人脸关键点。
    try:
        # 获取人脸关键点。
        preds = face_align.get_landmarks_from_image(input_img)
    except Exception as e:
        print("Error detecting face:", file_name)
        return None
    # 未检测到图片中含有人脸
    if preds is None:
        print("No face detected:", file_name)
        return None

    #计算各区域边界。
    left_eye_left = math.ceil(preds[0][36, 0] - (preds[0][39, 0] - preds[0][36, 0])*0.1)
    left_eye_top = math.ceil(min(preds[0][37, 1], preds[0][38, 1]) - (max(preds[0][41, 1], preds[0][40, 1]) - min(preds[0][37, 1], preds[0][38, 1]))*0.1)
    left_eye_bottom = math.ceil(max(preds[0][41, 1], preds[0][40, 1]) + (max(preds[0][41, 1], preds[0][40, 1]) - min(preds[0][37, 1], preds[0][38, 1]))*0.1)
    right_eye_top = math.ceil(min(preds[0][43, 1], preds[0][44, 1]) - (max(preds[0][47, 1], preds[0][46, 1]) - min(preds[0][43, 1], preds[0][44, 1]))*0.1)
    right_eye_right = math.ceil(preds[0][45, 0] + (preds[0][45, 0] - preds[0][42, 0])*0.1)
    right_eye_bottom = math.ceil(max(preds[0][47, 1], preds[0][46, 1]) + (max(preds[0][47, 1], preds[0][46, 1]) - min(preds[0][43, 1], preds[0][44, 1]))*0.1)
    mouth_left = math.ceil(preds[0][48, 0] - (preds[0][54, 0] - preds[0][48, 0])*0.1)
    mouth_right = math.ceil(preds[0][54, 0] + (preds[0][54, 0] - preds[0][48, 0])*0.1)
    mouth_bottom = math.ceil(preds[0][57, 1] + (preds[0][57, 1] - min(preds[0][50, 1], preds[0][52, 1]))*0.1)

    #计算各个区域的掩码区域。
    mask_area1 = np.array([[[left_eye_left, left_eye_bottom], [0, left_eye_bottom], [0, 0], [left_eye_left, 0]]], dtype = np.int32)
    mask_area2 = np.array([[[right_eye_right, min(left_eye_top, right_eye_top)], [left_eye_left, min(left_eye_top, right_eye_top)], [left_eye_left, 0], [right_eye_right, 0]]], dtype = np.int32)
    mask_area3 = np.array([[[input_img.shape[:2][1], right_eye_bottom], [right_eye_right, right_eye_bottom], [right_eye_right, 0], [input_img.shape[:2][1], 0]]], dtype = np.int32)
    mask_area4 = np.array([[[mouth_left, mouth_bottom], [0, mouth_bottom], [0, left_eye_bottom], [mouth_left, left_eye_bottom]]], dtype = np.int32)
    mask_area5 = np.array([[[input_img.shape[:2][1], mouth_bottom], [mouth_right, mouth_bottom], [mouth_right, right_eye_bottom], [input_img.shape[:2][1], right_eye_bottom]]], dtype = np.int32)
    mask_area6 = np.array([[[mouth_left, input_img.shape[:2][0]], [0, input_img.shape[:2][0]], [0, mouth_bottom], [mouth_left, mouth_bottom]]], dtype = np.int32)
    mask_area7 = np.array([[[mouth_right, input_img.shape[:2][0]], [mouth_left, input_img.shape[:2][0]], [mouth_left, mouth_bottom], [mouth_right, mouth_bottom]]], dtype = np.int32)
    mask_area8 = np.array([[[input_img.shape[:2][1], input_img.shape[:2][0]], [mouth_right, input_img.shape[:2][0]], [mouth_right, mouth_bottom], [input_img.shape[:2][1], mouth_bottom]]], dtype = np.int32)
    mask_list = [mask_area1, mask_area2, mask_area3, mask_area4, mask_area5, mask_area6, mask_area7, mask_area8]

    random.shuffle(mask_list)
    if mask_method == 'black': #随机选择6个区域，使用黑色填充。
        masked = input_img
        for mask_area in mask_list[0:2]:#超参数：3
            mask = np.full(input_img.shape[:2], 255, dtype = 'uint8')
            cv2.polylines(mask, mask_area, 1, 255)
            cv2.fillPoly(mask, mask_area, 0)
            masked = cv2.bitwise_and(masked, masked, mask=mask)
        #cv2.imshow('masked', masked)
    elif mask_method == 'noise': #随机选择6个区域，添加高斯噪声。
        masked = np.array(input_img).copy()
        masked = masked / 255.0
        for mask_area in mask_list[0:2]:#超参数：3
            mask_shape = (mask_area[0][0][1] - mask_area[0][2][1], mask_area[0][0][0] - mask_area[0][2][0], 3)
            # 产生高斯 noise
            noise = np.random.normal(0, 1, mask_shape)
            # 将噪声和图片叠加
            masked[mask_area[0][2][1]:mask_area[0][0][1], mask_area[0][2][0]:mask_area[0][0][0]] = masked[mask_area[0][2][1]:mask_area[0][0][1], mask_area[0][2][0]:mask_area[0][0][0]] + noise
            # 将超过 1 的置 1，低于 0 的置 0
            masked = np.clip(masked, 0, 1)
        # 将图片灰度范围的恢复为 0-255
        masked = np.uint8(masked*255)
        #cv2.imshow('masked', masked)

    return masked

# 人脸关键点定位
def get_face_align(dev):
    return face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=dev)

# 读取文件夹中的图片，并生成对应的掩码处理后的结果。
def process_images_in_folder(folder_path, fa, save_path, mask_method):
    # 获取文件夹内所有文件名
    files = os.listdir(folder_path)

    # 创建 tqdm 进度条
    progress_bar = tqdm(total=len(files), desc="Processing images")

    # 遍历文件夹内的所有文件
    for file_name in files:
        # 构建文件路径
        file_path = os.path.join(folder_path, file_name)

        # 如果是图片文件
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            # 读取图片
            input_img = cv2.imread(file_path)

            # 处理图片
            masked = get_masked_face(input_img, fa, mask_method, file_name)

            # 如果检测到了人脸并处理mask
            if masked is not None:
                # 保存处理后的图片
                output_path = os.path.join(save_path, file_name.split('.')[0] + '_masked.png')
                cv2.imwrite(output_path, masked)
            # 更新进度条
            progress_bar.update(1)

    # 关闭进度条
    progress_bar.close()


if __name__ == '__main__':
    # 图片文件夹位置\
    folder_path = '/root/autodl-tmp/FF++/ff_face_data/youtube'
    # 处理过的图片保存位置
    save_path = '/root/autodl-tmp/FF++/ff_maskedface_data/youtube'
    # 掩码方法
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    mask_method = 'black' # 'black', 'noise' or 'none'
    # 人脸关键点定位所用设备
    fa = get_face_align('cuda')
    start = time.time()
    process_images_in_folder(folder_path, fa, save_path, mask_method)



    end = time.time()
    total = end - start
    print("spend time: {} m {} s".format(int(total / 60), int(total % 60)))