import os
import pickle
def compress_txt_to_pkl(txt_file_path, pkl_file_path):
    """
    将单个txt文件的内容以字典形式压缩为pkl文件
    :param txt_file_path: 输入txt文件路径
    :param pkl_file_path: 输出pkl文件路径
    """
    data_dict = {}
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                key, value = parts
                data_dict[key] = value
    
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Compressed {txt_file_path} to {pkl_file_path}")

def compress_multiple_txt_to_pkl(txt_file_paths, pkl_file_path):
    """
    将多个txt文件的内容以字典形式合并并压缩为一个pkl文件
    :param txt_file_paths: 输入txt文件路径列表
    :param pkl_file_path: 输出pkl文件路径
    """
    combined_data = {}
    
    for txt_file_path in txt_file_paths:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    key, value = parts
                    combined_data[key] = value
    
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(combined_data, f)
    
    print(f"Compressed {len(txt_file_paths)} files to {pkl_file_path}")


# 示例使用
single_txt_file = 'CROHME/16_test_labels.txt'
single_pkl_file = 'CROHME/16_test_labels.pkl'
compress_txt_to_pkl(single_txt_file, single_pkl_file)

multiple_txt_files = ['example1.txt', 'example2.txt', 'example3.txt']
combined_pkl_file = 'combined.pkl'
# (multiple_txt_files, combined_pkl_file)
