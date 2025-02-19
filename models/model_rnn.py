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

"""
# Model for agent
Crucial model regarding how you set up your agent's neural network

- We suggest you prerpare enough layers between present state and next state.
- We suggest you prerpare enough layers between state and reward.
- We suggest you prerpare enough layers between state and action.
- In our experience, how the neural net of the agent handles the information flow toward reward will have immense impact on the performance of the agent. 
"""

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

        neural_types = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }
        self.bidirectional        = False
        self.recurrent_layers     = neural_types[self.neural_type.lower()](self.feature_size, self.feature_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=self.bidirectional)
        
        self.reward_linear        = nn.Linear(self.feature_size, self.reward_size , bias=self.bias)
        self.state_linear_        = nn.Linear(self.feature_size, self.state_size  , bias=self.bias)

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
            'mean_squared_error': torch.nn.MSELoss(),
            'binary_crossentropy': torch.nn.BCELoss()
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

        window_list   = list()
        
        for i in range(history_s.size(1)):
            s  = self.state_linear(history_s[:, i].unsqueeze(1))
            window_list.append(s)
            a  = self.action_linear(history_a[:,i].unsqueeze(1))
            window_list.append(a)

        s  = self.state_linear(present_s.unsqueeze(1))
        window_list.append(s)

        for i in range(future_a.size(1)):

            a  = self.action_linear(future_a[:,i].unsqueeze(1))
            window_list.append(a)

            h  = torch.cat(window_list, dim=1)
            h  = torch.tanh(h)
            
            """
            RNN, GRU, LSTM
            """
            h, _ = self.recurrent_layers(h)
            
            """
            We utilize the last idx in h to derive the latest reward and state.
            """
            r  = self.reward_linear(h[:, - 1, :])   
            r  = torch.sigmoid(r)
            s  = self.state_linear_(h[:, - 1, :])   
            s  = torch.tanh(s)

            future_r_list.append(r)
            future_s_list.append(s)

            """
            We save the latest state into the next round or time step.
            """
            s  = self.state_linear(s.unsqueeze(1))
            window_list.append(s)

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
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                initializer(layer.weight)
