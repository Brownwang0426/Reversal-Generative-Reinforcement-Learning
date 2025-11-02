
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
from tqdm.auto import tqdm
from collections import defaultdict

import itertools

import dill

import warnings
warnings.filterwarnings('ignore')

import concurrent.futures
import hashlib




class state_activation(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x 




class build_model(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 reward_size,
                 feature_size,
                 history_size,
                 future_size,
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
        self.history_size         = history_size
        self.future_size          = future_size
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
        self.state_norm           = nn.LayerNorm(self.feature_size, elementwise_affine=True)
        self.action_norm          = nn.LayerNorm(self.feature_size, elementwise_affine=True)
        self.dropout_0            = nn.Dropout(self.drop_rate)

        neural_types = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }
        self.bidirectional        = False
        self.recurrent_layers     = neural_types[self.neural_type.lower()](self.feature_size, self.feature_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=self.bidirectional)

        self.dropout_1            = nn.Dropout(self.drop_rate)
        self.reward_linear        = nn.Linear(self.feature_size, self.reward_size, bias=self.bias)
        self.state_linear_        = nn.Linear(self.feature_size, self.state_size , bias=self.bias)

        self.state_activate       = state_activation()

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
            'mean_squared_error': torch.nn.MSELoss(reduction='mean'),
            'binary_crossentropy': torch.nn.BCELoss(reduction='mean')
        }
        self.loss_function = losses[self.loss .lower()]

        # Loss function
        losses = {
            'mean_squared_error': torch.nn.MSELoss(reduction='none'),
            'binary_crossentropy': torch.nn.BCELoss(reduction='none')
        }
        self.loss_function_ = losses[self.loss .lower()]




    def forward(self, history_s, history_a, present_s, future_s, future_a):

        future_r_list = list()
        future_s_list = list()




        if history_s.size(1) > 0:
            history_s = self.state_norm (self.state_linear (history_s) )
            history_a = self.action_norm(self.action_linear(history_a) )
        present_s = self.state_norm (self.state_linear (present_s.unsqueeze(1)))
        future_a  = self.action_norm(self.action_linear(future_a              ))

        window_list   = list()
        if history_s.size(1) > 0:
            for i in range(history_s.size(1)):
                window_list.append(history_s[:, i:i+1] + history_a[:, i:i+1]) 
        


        
        for i in range(future_a.size(1)):

            window_list.append(present_s + future_a[:, i:i+1])
            h = torch.cat(window_list, dim=1)
            h = F.gelu(h)
            h = self.dropout_0(h)

            """
            RNN, GRU, LSTM
            """
            h, _ = self.recurrent_layers(h)
            """
            RNN, GRU, LSTM
            """

            h = self.dropout_1(h)
            r = self.reward_linear(h[:, -1:, :])
            r = torch.tanh(r)  
            s = self.state_linear_(h[:, -1:, :])
            s = self.state_activate(s)
            
            future_r_list.append(r)
            future_s_list.append(s)

            present_s = s
            present_s = self.state_norm(self.state_linear(present_s)) 

        future_r = torch.cat(future_r_list, dim=1) # future_r becomes [batch_size, sequence_size, reward_size]
        future_s = torch.cat(future_s_list, dim=1) # future_s becomes [batch_size, sequence_size, state_size ]
    
        return future_r, future_s




    def _forward(self, history_s, history_a, present_s, future_s, future_a):
    
        future_r_list = list()
        future_s_list = list()
    
    
    
    
        if history_s.size(1) > 0:
            history_s = self.state_norm (self.state_linear (history_s) )
            history_a = self.action_norm(self.action_linear(history_a) )
        present_s = self.state_norm (self.state_linear (present_s.unsqueeze(1)))
        future_a  = self.action_norm(self.action_linear(future_a              ))
    
        if history_s.size(1) > 0:
            history_s_a = history_s + history_a
        else:
            history_s_a = torch.empty((present_s.size(0), 0, present_s.size(2)), device=present_s.device, dtype=present_s.dtype)
                
    


        hidden_cache = None
    
        for i in range(future_a.size(1)):
    
            if i == 0:
                h = torch.cat([history_s_a, (present_s + future_a[:, i:i+1])], dim=1)
            else:
                h = present_s + future_a[:, i:i+1]
            h = F.gelu(h)
            h = self.dropout_0(h)
    
            """
            Transformer decoder
            """
            h, hidden_cache = self.recurrent_layers(h, hidden_cache)
            """
            Transformer decoder
            """
    
            h = self.dropout_1(h)
            r = self.reward_linear(h[:, -1:, :])
            r = torch.tanh(r)  
            s = self.state_linear_(h[:, -1:, :])
            s = self.state_activate(s)
    
            future_r_list.append(r)
            future_s_list.append(s)
    
            present_s = s
            present_s = self.state_norm(self.state_linear(present_s)) 
    
        future_r = torch.cat(future_r_list, dim=1) # future_r becomes [batch_size, sequence_size, reward_size]
        future_s = torch.cat(future_s_list, dim=1) # future_s becomes [batch_size, sequence_size, state_size ]
    
        return future_r, future_s



    
    def forward_(self, history_s, history_a, present_s, future_s, future_a):




        if history_s.size(1) > 0:
            history_s = self.state_norm (self.state_linear (history_s) )
            history_a = self.action_norm(self.action_linear(history_a) )
        present_s = self.state_norm (self.state_linear (present_s.unsqueeze(1)))
        future_s_ = self.state_norm (self.state_linear (future_s)[:, :-1, :])
        future_a  = self.action_norm(self.action_linear(future_a) )

        if history_s.size(1) > 0:
            history_s_a = history_s + history_a
        else:
            history_s_a = torch.empty((present_s.size(0), 0, present_s.size(2)), device=present_s.device, dtype=present_s.dtype)
                



        future_s_a = torch.cat((present_s, future_s_), dim=1) + future_a
        h = torch.cat([history_s_a, future_s_a], dim=1)
        h = F.gelu(h)
        h = self.dropout_0(h)

        """
        RNN, GRU, LSTM
        """
        h, _ = self.recurrent_layers(h)
        """
        RNN, GRU, LSTM
        """

        h = self.dropout_1(h)
        r = self.reward_linear(h)
        r = torch.tanh(r)  
        s = self.state_linear_(h)
        s = self.state_activate(s)

        future_r = r[:, -future_a.size(1):, :]
        future_s = s[:, -future_a.size(1):, :] 

        return future_r, future_s
    



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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                initializer(module.weight)









