import torch
import torch
from collections import OrderedDict
from models.Backbone import Backbone
from utils import load_config, load_checkpoint
from dataset import Words
import argparse
import random
import numpy as np


params = load_config('all_codes/weight_test/config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device
words = Words('data/word.txt')
params['word_num'] = len(words)
params['struct_num'] = 7
params['words'] = words




"""random seed"""
random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
torch.cuda.manual_seed(params['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
params['device'] = device



model = Backbone(params)

import torch
from collections import OrderedDict

def get_weight_structure(weights):
    def recursive_dict(structure, keys, value):
        if isinstance(value, torch.Tensor):
            if len(keys) == 1:
                structure[keys[0]] = f'[shape: {list(value.shape)}]'
            else:
                if keys[0] not in structure:
                    structure[keys[0]] = {}
                recursive_dict(structure[keys[0]], keys[1:], value)
        elif isinstance(value, OrderedDict):
            for sub_key, sub_value in value.items():
                recursive_dict(structure, keys + [sub_key], sub_value)

    weight_structure = {}
    for key, value in weights.items():
        keys = key.split('.')
        recursive_dict(weight_structure, keys, value)

    return weight_structure

def save_to_file(data, filename):
    import json
    with open(filename, 'a') as f:
        json.dump(data, f, indent=2)

def main():
    pth_file = '/Users/yejieguo/Desktop/san/checkpoints/SAN_decoder/best.pth'
    output_file = 'dict_weight.txt'

    
    #weights = torch.load(pth_file, map_location=torch.device('cpu'))
    weights = model.state_dict()
    
    # Get the weight structure
    weight_structure = get_weight_structure(weights)

    # Save the structure to a file
    save_to_file(weight_structure, output_file)

if __name__ == "__main__":
    main()
