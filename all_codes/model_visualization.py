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
from torchviz import make_dot
import torch.nn as nn


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

        
class pic(nn.Module):

    def __init__(self, params):
        super(pic, self).__init__()

        self.params = params
        self.channel = params['encoder']['out_channels']
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']

        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        self.encoder_feature_conv = nn.Conv2d(self.channel, self.attention_dim, kernel_size=1)

        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)
        self.input_size = params['decoder']['input_size']#256
        self.hidden_size = params['decoder']['hidden_size']#256
        self.out_channel = params['encoder']['out_channels']#684
        self.word_num = params['word_num'] #115
        self.dropout_prob = params['dropout']
        self.device = params['device']
        self.struct_num = params['struct_num']
        self.struct_dict = [108, 109, 110, 111, 112, 113, 114] #结构词

        self.ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet']['conv1_stride']#16

        self.threshold = params['hybrid_tree']['threshold']

        # init hidden state
        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)

        # word embedding
        self.embedding = nn.Embedding(self.word_num, self.input_size)#115->256

        # word gru
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.word_out_gru = nn.GRUCell(self.out_channel, self.hidden_size)

        # structure gru
        self.struc_input_gru = nn.GRUCell(self.input_size, self.hidden_size)



        # state to word/struct
        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.word_embedding_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size // 2)
        self.word_convert = nn.Linear(self.hidden_size // 2, self.word_num)

        self.struct_convert = nn.Linear(self.hidden_size // 2, self.struct_num)

        """ child to parent """
        self.c2p_input_gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.c2p_out_gru = nn.GRUCell(self.out_channel, self.hidden_size)


        self.c2p_state_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.c2p_word_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.c2p_relation_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.c2p_context_weight = nn.Linear(self.out_channel, self.hidden_size // 2)
        self.c2p_convert = nn.Linear(self.hidden_size // 2, self.word_num)
        ############################################################################################################
        self.hidden_state = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.c2p_hidden_state = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.final_fusion = nn.Linear(self.hidden_size // 2 * 4, self.hidden_size // 2)
        self.c2pfinal_fusion = nn.Linear(self.hidden_size // 2 * 5, self.hidden_size // 2)
        ############################################################################################################
        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])
            
    def init_hidden(self, features, feature_mask):

                average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
                average = self.init_weight(average)

                return torch.tanh(average)
           

    def forward(self, cnn_features, images_mask):

        height, width = cnn_features.shape[2:]#cnn_features_shape= torch.Size([1, 684, 12, 40])
        # images_mask_shape= torch.Size([1, 1, 181, 635]) 根据实际图片的像素变化，除以16变成最后两维的数字
        
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]#images_mask_shape= torch.Size([1, 1, 12, 40]
    

        word_alpha_sum = torch.zeros((1, 1, height, width)).to(device=self.device)
        struct_alpha_sum = torch.zeros((1, 1, height, width)).to(device=self.device)

        if False:
            pass

        else:
            word_embedding = self.embedding(torch.ones(1).long().to(device=self.device))
            struct_list = []
            parent_hidden = torch.randn(1, self.hidden_size)

            prediction = ''
            right_brace = 0
            cid, pid = 0, 0
            p_re = 'Start'
            word = torch.LongTensor([1]) # 1
            result = [['<s>', 0, -1, 'root']]

            while len(prediction) < 400:

                # word
                # shape_of_word_embedding= torch.Size([1, 256])
                # shape of parent_hidden= torch.Size([1, 256])
                word_hidden_first = self.word_input_gru(word_embedding, parent_hidden)# shape of word_hidden_first =  torch.Size([1, 256])

                query = self.hidden_weight(word_hidden_first) #256->512
        
                alpha_sum_trans = self.attention_conv(word_alpha_sum)#
                coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0,2,3,1))

                cnn_features_trans = self.encoder_feature_conv(cnn_features)

                alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha + cnn_features_trans.permute(0,2,3,1))
                energy = self.alpha_convert(alpha_score)
                energy = energy - energy.max()
                energy_exp = torch.exp(energy.squeeze(-1))
                alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)

                word_alpha_sum = alpha[:,None,:,:] + word_alpha_sum

                word_context_vec = (alpha[:,None,:,:] * cnn_features).sum(-1).sum(-1)
                
                
                
                # shape of word_context_vec =  torch.Size([1, 684])
                # shape of word_alpha_sum =  torch.Size([1, 1, 12, 40])
                hidden = self.word_out_gru(word_context_vec, word_hidden_first)# shape of hidden =  torch.Size([1, 256])
                
                current_state = self.word_state_weight(hidden)#shape of current_state =  torch.Size([1, 128])
                
                word_weighted_embedding = self.word_embedding_weight(word_embedding)#shape of word_weighted_embedding =  torch.Size([1, 128])
                
                word_context_weighted = self.word_context_weight(word_context_vec)#shape of word_context_weighted =  torch.Size([1, 128])
               
                hidden_state = self.hidden_state(word_hidden_first)#shape of hidden_state =  torch.Size([1, 128])
                
                word_out_state = self.final_fusion(torch.cat((current_state, word_weighted_embedding, word_context_weighted, hidden_state), 1))#shape of word_out_state =  torch.Size([1, 128])
                
                if self.params['dropout']:
                    word_out_state = self.dropout(word_out_state)#shape of word_out_state =  torch.Size([1, 128])
                    
                    

                word_prob = self.word_convert(word_out_state)#shape of word_prob =  torch.Size([1, 115])
                
                return word_prob
            
            
            
model = pic(params)
x = torch.randn(1, 684, 6, 29)  # 假设输入是 (batch_size, seq_length, input_size)
mask = torch.randn(1, 684, 6, 29)
# 获取模型输出
output = model(x, mask)

# 生成计算图
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('model_visualization')