
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
        # self.W_q           = nn.Linear(feature_size, feature_size, bias=self.bias)
        # self.W_k           = nn.Linear(feature_size, feature_size, bias=self.bias)
        # self.W_v           = nn.Linear(feature_size, feature_size, bias=self.bias)
        # self.W_o           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.attn_dropout  = nn.Dropout(self.drop_rate)
        self.resid_dropout = nn.Dropout(self.drop_rate)

        self.neural_type = 'lstm'
        neural_types = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }
        self.W_q     = neural_types[self.neural_type.lower()](self.feature_size, self.feature_size, num_layers=1, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=False)
        self.W_k     = neural_types[self.neural_type.lower()](self.feature_size, self.feature_size, num_layers=1, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=False)
        self.W_v     = neural_types[self.neural_type.lower()](self.feature_size, self.feature_size, num_layers=1, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=False)
        self.W_o     = neural_types[self.neural_type.lower()](self.feature_size, self.feature_size, num_layers=1, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=False)
        
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
        Q    = self.split_heads(self.W_q(Q)[0])  # Shape: (batch_size, num_heads, sequence_size, head_size )
        K    = self.split_heads(self.W_k(K)[0])  # Shape: (batch_size, num_heads, sequence_size, head_size )
        V    = self.split_heads(self.W_v(V)[0])  # Shape: (batch_size, num_heads, sequence_size, head_size )
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output      = self.W_o(self.combine_heads(attn_output))[0]
        output      = self.resid_dropout(output)
        return output




class build_model(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 reward_size,
                 feature_size,
                 sequence_size,
                 neural_type,
                 num_layers,
                 num_heads,
                 init,
                 opti,
                 loss,
                 bias,
                 drop_rate,
                 alpha):

        super(build_model, self).__init__()

        self.state_size           = state_size
        self.action_size          = action_size
        self.reward_size          = reward_size
        self.feature_size         = feature_size
        self.sequence_size        = sequence_size
        self.neural_type          = neural_type
        self.num_layers           = num_layers
        self.num_heads            = num_heads
        self.init                 = init
        self.opti                 = opti
        self.loss                 = loss
        self.bias                 = bias
        self.drop_rate            = drop_rate
        self.alpha                = alpha

        self.state_linear         = nn.Linear(self.state_size  , self.feature_size, bias=self.bias)
        self.action_linear        = nn.Linear(self.action_size , self.feature_size, bias=self.bias)

        self.positional_encoding  = nn.Parameter(self.generate_positional_encoding(self.sequence_size, self.feature_size ), requires_grad=False)
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
        mask                      = torch.full((1, 1, self.sequence_size, self.sequence_size), float("-inf"))
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
        losses = {
            'mean_squared_error': torch.nn.MSELoss(reduction='sum'),
            'binary_crossentropy': torch.nn.BCELoss(reduction='sum')
        }
        self.loss_function = losses[self.loss .lower()]

        # Loss function
        losses = {
            'mean_squared_error': torch.nn.MSELoss(reduction='none'),
            'binary_crossentropy': torch.nn.BCELoss(reduction='none')
        }
        self.loss_function_ = losses[self.loss .lower()]




    def forward(self, history_s, history_a, present_s, future_a):

        future_r_list = list()
        future_s_list = list()




        if history_s.size(1) > 0:
            history_s = self.state_linear (history_s)  
            history_a = self.action_linear(history_a) 
        present_s = self.state_linear (present_s.unsqueeze(1))
        future_a  = self.action_linear(future_a) 

        if history_s.size(1) > 0:
            history_s_a = history_s + history_a
        else:
            history_s_a = torch.empty((present_s.size(0), 0, present_s.size(2)), device=present_s.device, dtype=present_s.dtype)
        
        for i in range(future_a.size(1)):

            history_s_a =  torch.cat([history_s_a, (present_s + future_a[:, i:i+1])], dim=1)

            h  = torch.tanh(history_s_a)

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
            r = torch.tanh(r)
            s = self.state_linear_(h[:, - 1, :])   
            s = torch.tanh(s)

            future_r_list.append(r)
            future_s_list.append(s)

            present_s = s
            present_s = self.state_linear(present_s.unsqueeze(1))

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
