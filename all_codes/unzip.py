import os
import pickle
import cv2
import numpy as np
import pickle

def convert_pkl_to_txt(pkl_file, txt_file):
    # 打开.pkl文件并加载数据
    with open(pkl_file, 'rb') as pkl_f:
        data = pickle.load(pkl_f)
    
    # 将数据写入.txt文件
    with open(txt_file, 'w', encoding='utf-8') as txt_f:
        txt_f.write(str(data))

# 示例使用


def extract_images_from_pkl(pkl_file, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取 pkl 文件
    with open(pkl_file, 'rb') as f:
        images_dict = pickle.load(f)

    # 遍历并保存图像
    for image_name, image_data in images_dict.items():
        # 假设图像数据是 numpy 数组
        if isinstance(image_data, np.ndarray):
            image_path = os.path.join(output_dir, f"{image_name}.png")
            cv2.imwrite(image_path, image_data)
        else:
            print(f"Unsupported data type for {image_name}: {type(image_data)}")

# 示例用法
# pkl_file = '/home/yjguo/SAN-main/CROHME/train_images.pkl'
# output_dir = os.path.splitext(pkl_file)[0]  # 使用 pkl 文件名作为文件夹名
# extract_images_from_pkl(pkl_file, output_dir)

pkl_file = 'data/train/test_label.pkl'
txt_file = 'data/train/test_label.txt'
convert_pkl_to_txt(pkl_file, txt_file)