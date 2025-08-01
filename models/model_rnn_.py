
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




class DeterministicDropout(nn.Module):
    def __init__(self, p=0.5):
        super(DeterministicDropout, self).__init__()
        self.p = p
        self.drop_mask = None
        self.locked = False

    def forward(self, x):
        if not self.locked:
            self.drop_mask = (torch.rand_like(x.sum(dim=tuple(range(x.ndim - 1)))) > self.p).float()
        return x * self.drop_mask

    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False




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
        self.dropout_0            = DeterministicDropout(self.drop_rate)

        neural_types = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }
        self.bidirectional        = False
        self.recurrent_layers     = neural_types[self.neural_type.lower()](self.feature_size, self.feature_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=self.bidirectional)

        self.dropout_1            = DeterministicDropout(self.drop_rate)
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

            h = torch.tanh(history_s_a)
            h = self.dropout_0(h)

            """
            RNN, GRU, LSTM
            """
            h, _ = self.recurrent_layers(h)

            """
            We utilize the last idx in h to derive the latest reward and state.
            """
            h = self.dropout_1(h)
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

    def lock(self):
        for module in self.modules():
            if isinstance(module, DeterministicDropout):
                module.lock()

    def unlock(self):
        for module in self.modules():
            if isinstance(module, DeterministicDropout):
                module.unlock()










