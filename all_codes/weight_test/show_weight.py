import torch
from collections import OrderedDict
from models.Backbone import Backbone
from utils import load_config, load_checkpoint
from dataset import Words

params = load_config('all_codes/weight_test/config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device
words = Words('data/word.txt')
params['word_num'] = len(words)
params['struct_num'] = 7
params['words'] = words

def print_weights_pseudocode(weights, prefix='', file=None):
    """
    递归遍历权重字典，打印字典结构，替换实际权重内容为 [shape:[...]]。
    统计并返回总的层数。
    """
    layer_count = 0

    def recurse(weights, prefix=''):
        nonlocal layer_count
        if isinstance(weights, OrderedDict):
            for key, value in weights.items():
                if isinstance(value, OrderedDict):
                    line = f"{prefix}{key}: {{\n"
                    if file:
                        file.write(line)
                    else:
                        print(line)
                    recurse(value, prefix + '  ')
                    line = f"{prefix}}},\n"
                    if file:
                        file.write(line)
                    else:
                        print(line)
                else:
                    line = f"{prefix}{key}: [shape:{list(value.shape)}],\n"
                    if file:
                        file.write(line)
                    else:
                        print(line)
                    layer_count += 1
        elif isinstance(weights, dict):
            for key, value in weights.items():
                if isinstance(value, (OrderedDict, dict)):
                    line = f"{prefix}{key}: {{\n"
                    if file:
                        file.write(line)
                    else:
                        print(line)
                    recurse(value, prefix + '  ')
                    line = f"{prefix}}},\n"
                    if file:
                        file.write(line)
                    else:
                        print(line)
                else:
                    line = f"{prefix}{key}: [shape:{list(value.shape)}],\n"
                    if file:
                        file.write(line)
                    else:
                        print(line)
                    layer_count += 1

    recurse(weights, prefix)
    return layer_count

# 加载 .pth 文件
pth_file = '/home/yjguo/san/train_ckpts/SAN_2024-05-26-15-50_Encoder-DenseNet_Decoder-SAN_decoder_max_size-320-1600/SAN_2024-05-26-15-50_Encoder-DenseNet_Decoder-SAN_decoder_max_size-320-1600_WordRate-0.9326_structRate-0.9929_ExpRate-0.7250_10.pth'
#weights = torch.load(pth_file, map_location=device)

model = Backbone(params)
weights = model.state_dict()

# 打开文件以写入模式
with open('all_codes/weight_test/weight.txt', 'a') as file:
    file.write("{\n")
    total_layers = print_weights_pseudocode(weights, prefix='  ', file=file)
    file.write("}\n")
    file.write('total_layers: ' + str(total_layers))
    file.write('\n')
    

print(f"Total number of layers: {total_layers}")
