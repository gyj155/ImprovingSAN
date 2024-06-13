import os
import time
import argparse
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils import load_config, save_checkpoint, load_checkpoint, my_load_checkpoints
from dataset import get_dataset
from models.Backbone import Backbone
from training import train, eval
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='HYB Tree')
parser.add_argument('--config', default='all_codes/config.yaml', type=str, help='path to config file')
parser.add_argument('--check', action='store_true', help='only for code check')
args = parser.parse_args()

if not args.config:
    print('please provide config yaml')
    exit(-1)

"""config"""
params = load_config(args.config)

"""random seed"""
random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
torch.cuda.manual_seed(params['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
params['device'] = device

train_loader, eval_loader = get_dataset(params)

model = Backbone(params)
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = f'{params["experiment"]}_{now}_Encoder-{params["encoder"]["net"]}_Decoder-{params["decoder"]["net"]}_' \
             f'max_size-{params["image_height"]}-{params["image_width"]}'
print(model.name)
model = model.to(device)

optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))

if params['finetune']:

    print('loading pretrain model weight')
    print(f'pretrain model: {params["checkpoint"]}')
    load_checkpoint(model, optimizer, params['checkpoint'])
    #my_load_checkpoints(model, torch.load(params['checkpoint'], map_location=device))

if not args.check:
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
        os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    os.system(f'cp {args.config} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')


min_score = 0
min_step = 0
for epoch in range(params['epoches']):
    print('training')
    train_loss, train_word_score, train_node_score, train_expRate = train(params, model, optimizer, epoch, train_loader, writer=0)
    print(f'Epoch: {epoch+1}  loss: {train_loss:.4f}  word score: {train_word_score:.4f}  struct score: {train_node_score:.4f} ')
    loss, word_right, struct_right, exp_right = eval(params, model, epoch, eval_loader, writer=0)
    print(f'Epoch: {epoch+1}  loss: {loss:.4f}  word score: {word_right:.4f}  struct score: {struct_right:.4f}  ExpRate: {exp_right:.4f}')
    if (epoch+1)%1 == 0 :
        save_checkpoint(model, optimizer, word_right, struct_right, exp_right, epoch+1, optimizer_save=params['optimizer_save'], path='train_ckpts')
   









