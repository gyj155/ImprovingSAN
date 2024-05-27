import torch
import torch.nn as nn
import sys
#from infer.Backbone import Backbone
from models.Backbone import Backbone
sys.append('..')
from utils import load_config, load_checkpoint, my_load_checkpoints
from dataset import Words




params = load_config('all_codes/weight_test/config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device
words = Words('data/word.txt') #words.words_dict = {words[i].strip(): i for i in range(len(words))}
params['word_num'] = len(words)
params['struct_num'] = 7
params['words'] = words

pth_file = 'checkpoints/SAN_decoder/best.pth'
model = Backbone(params)

my_load_checkpoints(model, torch.load(pth_file, map_location=device))

# model.eval()
# image = torch.randn(1, 1, 181, 635)
# image = (image - image.min()) / (image.max() - image.min()) 
# image = image * 255  # 扩展到0-255范围
# image_mask = torch.ones(1, 1, 181, 635)

# prediction = model(image, image_mask)

# print("params matched")