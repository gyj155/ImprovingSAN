import os
import cv2
import argparse
import torch
import json
from tqdm import tqdm
import sys
from utils import load_config, load_checkpoint
from infer.Backbone import Backbone
from dataset import Words
import numpy as np

parser = argparse.ArgumentParser(description='Spatial channel attention')
parser.add_argument('--config', default='all_codes/config.yaml', type=str)
args = parser.parse_args()
image_paths = ["data/CROHME/14_test_images","data/CROHME/16_test_images","data/CROHME/19_test_images"]
label_paths = ["data/CROHME/14_test_labels.txt","data/CROHME/16_test_labels.txt","data/CROHME/19_test_labels.txt"]
if not args.config:
    print('config file is needed')
    exit(-1)

params = load_config(args.config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device
words = Words(params['word_path'])
params['word_num'] = len(words)
params['struct_num'] = 7
params['words'] = words

model = Backbone(params)
model = model.to(device)
optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))

#load_checkpoint(model, None, params['checkpoint'])
load_checkpoint(model, optimizer, 'SAN_2024-05-27-08-51_Encoder-DenseNet_Decoder-SAN_decoder_max_size-320-1600_WordRate-0.8733_structRate-0.9859_ExpRate-0.4888_1.pth')

model.eval()

def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3]]) #从父节点产生的above/below/right
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Above':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Below':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['Right', 'Above', 'Below']:
                    return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['l_sup']:
                    return_string += ['['] + convert(child_list[i][1], gtd_list) + [']']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Inside':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sub', 'Below']:
                    return_string += ['_', '{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sup', 'Above']:
                    return_string += ['^', '{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)
        return return_string

for img_path, label_path in zip(image_paths, label_paths):
    word_right, node_right, exp_right, length, cal_num = 0, 0, 0, 0, 0
    with open(label_path) as f:
        labels = f.readlines()


    with torch.no_grad():      
        for item in tqdm(labels):
            name, *label = item.split()
            label = ' '.join(label)
            img = cv2.imread(os.path.join(img_path, name + '.png'))
            if img is None:
                img = cv2.imread(os.path.join(img_path, name + '_0.bmp')) 
            if img is None:
                print(f"Reading {name} failed")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = img.shape
            if h > 1500 or w > 1500:
                scale = 1500 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape

            if h < 80 or w < 80:
                if h < 80:
                    pad_h = (256 - h) // 2
                    img = np.pad(img, ((pad_h, 256 - h - pad_h), (0, 0)), mode='constant', constant_values=0)
                    h = 256
                if w < 80:
                    pad_w = (256 - w) // 2
                    img = np.pad(img, ((0, 0), (pad_w, 256 - w - pad_w)), mode='constant', constant_values=0)
                    w = 256
            image = torch.Tensor(img) / 255
            image = image.unsqueeze(0).unsqueeze(0)
            image_mask = torch.ones(image.shape)
            image, image_mask = image.to(device), image_mask.to(device)
            prediction = model(image, image_mask)
            print('prediction:',prediction)
            latex_list = convert(1, prediction)
            latex_string = ' '.join(latex_list)
            print('final_string:',latex_string)
            if latex_string == label.strip():
                exp_right += 1         
            else:
                bad_case = {
                    'name': name,
                    'label': label,
                    'predi': latex_string,
                    'list': prediction
                }
                with open('all_codes/test_bad_case.jsonl', 'a') as f:
                    json.dump(bad_case, f, ensure_ascii=False)
                    f.write('\n')
            #sys.exit()

        print(f'Test name: {img_path.split("/")[-1]}\nacc:{exp_right / len(labels)}')
