
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import minigrid

import numpy as np
import math
from scipy.special import softmax

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, TensorDataset, Subset

import csv

import multiprocessing as mp
import os
import sys
import copy
import random
import gc
import time
from tqdm import tqdm
from collections import defaultdict

import itertools

import dill

import warnings
warnings.filterwarnings('ignore')

import concurrent.futures
import hashlib


class custom_attn(nn.Module):
    def __init__(self, feature_size, num_heads, bias, drop_rate):
        super(custom_attn, self).__init__()
        assert feature_size % num_heads == 0, "feature_size must be divisible by num_heads"
        self.feature_size  = feature_size
        self.num_heads     = num_heads
        self.head_size     = feature_size // num_heads
        self.bias          = bias
        self.drop_rate     = drop_rate
        self.W_q           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_k           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_v           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_o           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.attn_dropout  = nn.Dropout(self.drop_rate)
        self.resid_dropout = nn.Dropout(self.drop_rate)

    def split_heads(self, x):
        batch_size, sequence_size, feature_size = x.size()
        return x.view(batch_size, sequence_size, self.num_heads, self.head_size).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5) #  (batch_size, num_heads, sequence_size, head_size) @ (batch_size, num_heads, head_size, sequence_size ) 
        
        if mask != None:
            attn_scores += mask                   # (batch_size, num_heads, sequence_size, sequence_size) += (batch_size, 1, sequence_size, sequence_size)
        else:
            attn_scores += 0

        attn_probs = torch.softmax(attn_scores, dim=-1) 
        attn_probs = self.attn_dropout (attn_probs)
        output     = torch.matmul(attn_probs, V)  # (batch_size, num_heads, sequence_size, sequence_size) @ (batch_size, num_heads, sequence_size, head_size ) 
        return output                             # (batch_size, num_heads, sequence_size, head_size)

    def combine_heads(self, x):
        batch_size, num_heads, sequence_size, head_size = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_size, self.feature_size)

    def forward(self, Q, K, V, mask=None):
        # mask Shape: (batch_size, 1, sequence_size, sequence_size)
        # Q    Shape: (batch_size,    sequence_size, feature_size )
        Q    = self.split_heads(self.W_q(Q))  # Shape: (batch_size, num_heads, sequence_size, head_size )
        K    = self.split_heads(self.W_k(K))  # Shape: (batch_size, num_heads, sequence_size, head_size )
        V    = self.split_heads(self.W_v(V))  # Shape: (batch_size, num_heads, sequence_size, head_size )
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output      = self.W_o(self.combine_heads(attn_output))
        output      = self.resid_dropout(output)
        return output




class custom_loss(nn.Module):
    def __init__(self, loss_scale, sequence_size):
        super(custom_loss, self).__init__()
        self.loss_weight = nn.Parameter(torch.tensor([loss_scale ** i for i in range(sequence_size)]), requires_grad=False)

    def forward(self, output, target):
        output           = torch.sum(output, dim=2)  # Shape: (batch_size, sequence_size)
        target           = torch.sum(target, dim=2)  # Shape: (batch_size, sequence_size)
        loss             = torch.sum(((output - target) ** 2) * self.loss_weight)  # Shape: () scalar
        return loss # / output.shape[0] 
    
# class custom_loss(nn.Module):
#     def __init__(self, loss_scale, sequence_size):
#         super(custom_loss, self).__init__()
#         self.loss_weight = nn.Parameter(torch.tensor([loss_scale ** i for i in range(sequence_size)]), requires_grad=False)
# 
#     def forward(self, output, target):                     # Shape: (batch_size, sequence_size, feature_size)
#         loss = torch.sum((output - target) ** 2, dim = 2)  # Shape: (batch_size, sequence_size)
#         loss = torch.sum( loss * self.loss_weight )  
#         return loss # / output.shape[0] 




class build_model(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 reward_size,
                 reward_scale,
                 feature_size,
                 sequence_size,
                 neural_type,
                 num_layers,
                 num_heads,
                 init,
                 opti,
                 loss_scale,
                 bias,
                 drop_rate,
                 alpha):

        super(build_model, self).__init__()

        self.state_size           = state_size
        self.action_size          = action_size
        self.reward_size          = reward_size
        self.reward_scale         = reward_scale
        self.feature_size         = feature_size
        self.sequence_size        = sequence_size
        self.neural_type          = neural_type
        self.num_layers           = num_layers
        self.num_heads            = num_heads
        self.init                 = init
        self.opti                 = opti
        self.loss_scale           = loss_scale
        self.bias                 = bias
        self.drop_rate            = drop_rate
        self.alpha                = alpha

        self.state_linear         = nn.Linear(self.state_size  , self.feature_size, bias=self.bias)
        self.action_linear        = nn.Linear(self.action_size , self.feature_size, bias=self.bias)

        self.positional_encoding  = nn.Parameter(self.generate_positional_encoding(2 * self.sequence_size, self.feature_size ), requires_grad=False)
        self.transformer_layers   = \
        nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(self.feature_size, elementwise_affine=True),
                custom_attn(self.feature_size, self.num_heads, self.bias, self.drop_rate),
                nn.LayerNorm(self.feature_size, elementwise_affine=True),
                nn.Linear(self.feature_size, self.feature_size, bias=self.bias)
            ])
            for _ in range(self.num_layers)
        ])
        self.transformer_norm     = nn.LayerNorm(self.feature_size, elementwise_affine=True) 
        mask                      = torch.full((1, 1, self.sequence_size*2, self.sequence_size*2), float("-inf"))
        mask                      = torch.triu(mask , diagonal=1)
        self.register_buffer('mask', mask)  

        self.reward_linear        = nn.Linear(self.feature_size, self.reward_size  , bias=self.bias)
        self.state_linear_        = nn.Linear(self.feature_size, self.state_size   , bias=self.bias)

        # Initialize weights for fully connected layers
        self.initialize_weights(self.init  )

        # Optimizer
        optimizers = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        self.selected_optimizer = optimizers[self.opti.lower()](self.parameters(), lr=self.alpha)

        # Loss function
        self.loss_function = custom_loss(self.loss_scale, self.sequence_size)




    def forward(self, history_s, history_a, present_s, future_a):

        future_r_list = list()
        future_s_list = list()




        if history_s.size(1) > 0:
            history_s = self.state_linear (history_s)  
            history_a = self.action_linear(history_a) 
        present_s = self.state_linear (present_s.unsqueeze(1))
        future_a  = self.action_linear(future_a) 




        window_list   = list()

        if history_s.size(1) > 0:
            for i in range(history_s.size(1)):
                window_list.append(history_s[:, i:i+1]) 
                window_list.append(history_a[:, i:i+1]) 
        window_list.append(present_s)
        
        for i in range(future_a.size(1)):

            window_list.append(future_a[:, i:i+1])

            h  = torch.cat(window_list, dim=1)
            h  = torch.tanh(h)

            """
            Transformer decoder
            """
            long = h.size(1)
            h    = h + self.positional_encoding[:, :long, :]
            for layer in self.transformer_layers:
                attention_norm, attention_linear, fully_connected_norm, fully_connected_linear = layer
                h_  = attention_norm(h)
                h   = h + attention_linear(h_, h_, h_, self.mask[:, :, :long, :long])
                h_  = fully_connected_norm(h)
                h   = h + fully_connected_linear(h_)
            h  = self.transformer_norm(h)

            """
            We utilize the last idx in h to derive the latest reward and state.
            """
            r = self.reward_linear(h[:, - 1, :])  
            r = torch.sigmoid(r) * self.reward_scale
            s = self.state_linear_(h[:, - 1, :])   
            s = torch.tanh(s)

            future_r_list.append(r)
            future_s_list.append(s)

            present_s = s
            present_s = self.state_linear(present_s.unsqueeze(1))

            window_list.append(present_s)

        future_r = torch.stack(future_r_list, dim=0).transpose(0, 1) # future_r becomes [batch_size, sequence_size, reward_size]
        future_s = torch.stack(future_s_list, dim=0).transpose(0, 1) # future_s becomes [batch_size, sequence_size, state_size ]
    
        return future_r, future_s




    def generate_positional_encoding(self, sequence_size, feature_size):
        pe = torch.zeros(sequence_size,feature_size)
        for pos in range(sequence_size):
            for i in range(0,feature_size,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/feature_size)))
                if i + 1 < feature_size:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/feature_size)))
        return pe.unsqueeze(0)  # Shape: (1, sequence_size, feature_size)

    def initialize_weights(self, initializer):
        initializers = {
            'random_uniform': nn.init.uniform_,
            'random_normal': nn.init.normal_,
            'glorot_uniform': nn.init.xavier_uniform_,
            'glorot_normal': nn.init.xavier_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_
        }
        initializer = initializers[initializer.lower()]
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                initializer(layer.weight)
