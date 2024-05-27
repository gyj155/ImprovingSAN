import os
import cv2
import argparse
import torch
import json
from tqdm import tqdm
from datetime import datetime
from io import StringIO
import sys

from utils import load_config, load_checkpoint
from infer.Backbone import Backbone
from dataset import Words

parser = argparse.ArgumentParser(description='Spatial channel attention')
parser.add_argument('--config', default='config.yaml', type=str, help='配置文件路径')
parser.add_argument('--image_path', default='/home/yuanye/work/data/CROHME2014/14_off_image_test', type=str, help='测试image路径')
parser.add_argument('--label_path', default='/home/yuanye/work/data/CROHME2014/test_caption.txt', type=str, help='测试label路径')
args = parser.parse_args()

if not args.config:
    print('请提供config yaml路径！')
    exit(-1)

# 重定向标准输出流
# old_stdout = sys.stdout
# sys.stdout = mystdout = StringIO()

# 加载config文件
params = load_config(args.config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device

words = Words(params['word_path']) #words.words_dict = {words[i].strip(): i for i in range(len(words))}
params['word_num'] = len(words)
params['struct_num'] = 7
params['words'] = words

model = Backbone(params)
model = model.to(device)
#load_checkpoint(model, None, params['checkpoint'])
load_checkpoint(model, None, '/home/yjguo/san/train_ckpts/SAN_2024-05-27-03-53_Encoder-DenseNet_Decoder-SAN_decoder_max_size-320-1600/SAN_2024-05-27-03-53_Encoder-DenseNet_Decoder-SAN_decoder_max_size-320-1600_WordRate-0.8349_structRate-0.9756_ExpRate-0.5000_2.pth')
model.eval()

word_right, node_right, exp_right, length, cal_num = 0, 0, 0, 0, 0

with open(args.label_path) as f:
    labels = f.readlines()

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

with torch.no_grad():
    bad_case = {}
    good_case = {}
    for item in tqdm(labels):
        name, *label = item.split()
        label = ' '.join(label)
        if name.endswith('.jpg') or name.endswith('.png') or name.endswith('.bmp'):
            name = os.path.splitext(name)[0]  # 获取没有扩展名的文件名

        # 尝试读取 PNG 图像
        img = cv2.imread(os.path.join(args.image_path, name + '.png'))
        if img is None:
            # 如果没有 PNG 图像，则尝试读取 BMP 图像
            img = cv2.imread(os.path.join(args.image_path, name + '_0.bmp'))
        
        if img is None:
            print(f"图像 {name} 读取失败")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
        
        image = torch.Tensor(img) / 255
        
        image = image.unsqueeze(0).unsqueeze(0)
        print("image = ", image.shape)
        image_mask = torch.ones(image.shape)
        image, image_mask = image.to(device), image_mask.to(device)

        prediction = model(image, image_mask)
        #print("prediction = ", prediction)
        
        
        latex_list = convert(1, prediction)
        latex_string = ' '.join(latex_list)
        print("latex_string = ", latex_string.strip())
        if latex_string == label.strip():
            exp_right += 1
            good_case = {
                'name': name,
                'label': label,
                'predi': latex_string,
                'list': prediction
            }
            with open('test_good_case.jsonl', 'a') as f:
                json.dump(good_case, f, ensure_ascii=False)
                f.write('\n')
        else:
            bad_case = {
                'name': name,
                'label': label,
                'predi': latex_string,
                'list': prediction
            }
            with open('test_bad_case.jsonl', 'a') as f:
                json.dump(bad_case, f, ensure_ascii=False)
                f.write('\n')
        

        
            
            
            #sys.exit()

    print(exp_right / len(labels))

        

# 获取终端输出内容
# sys.stdout = old_stdout
# terminal_output = mystdout.getvalue()

# 创建或更新历史记录文件
history_file = 'history.json'
history_data = []

if os.path.exists(history_file):
    try:
        with open(history_file, 'r') as f:
            history_data = json.load(f)
    except json.JSONDecodeError:
        history_data = []

current_run = {
    'idx': len(history_data) + 1,
    'date': datetime.now().strftime('%Y-%m-%d'),
    'time': datetime.now().strftime('%H:%M:%S'),
    'config_path': args.config,
    'image_path': args.image_path,
    'label_path': args.label_path,
    #'terminal_output': terminal_output
}

history_data.append(current_run)

with open(history_file, 'w') as f:
    json.dump(history_data, f, ensure_ascii=False, indent=4)
