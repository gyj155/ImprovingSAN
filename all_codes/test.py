image_path = '/home/yjguo/san/data/CROHME/16_test_images'
image_out = '16_image.pkl'
label_path = 'train_hyb'
label_out = '16_label.pkl'
import glob
import cv2
import os
import pickle as pkl
from tqdm import tqdm




labels = glob.glob(os.path.join(label_path, '*.txt'))
print(labels)