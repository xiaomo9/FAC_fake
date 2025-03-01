import os
import json

"""
自制.json标签文件
"""


# 视频文件夹路径和输出的JSON文件路径
video_folder = r'E:\Daima\dataset\FF++\FF++\manipulated_sequences\Face2Face'
output_json = r'E:\Daima\dataset\FF++\FF++\metadate1.json'
custom_label = 'FAKE'  # 自定义标签，所有文件都使用该标签

# 获取文件夹中所有.mp4视频文件名
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

# 如果JSON文件存在，先加载现有数据；否则创建一个空字典
if os.path.exists(output_json):
    with open(output_json, 'r') as json_file:
        data = json.load(json_file)
else:
    data = {}

# 将新视频文件名和标签添加到现有数据中，避免重复
for video in video_files:
    if video not in data:  # 检查文件是否已存在，防止覆盖
        data[video] = {"label": custom_label}

# 保存更新后的数据到JSON文件
with open(output_json, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON文件已更新: {output_json}")
