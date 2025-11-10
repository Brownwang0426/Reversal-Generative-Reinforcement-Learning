
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
                 alpha,
                 L2_lambda,
                 grad_clip_value):

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
        self.L2_lambda            = L2_lambda
        self.grad_clip_value      = grad_clip_value

        self.state_linear         = nn.Linear(self.state_size  , self.feature_size, bias=self.bias)
        self.action_linear        = nn.Linear(self.action_size , self.feature_size, bias=self.bias)
        self.state_norm           = nn.LayerNorm(self.feature_size, elementwise_affine=True)
        self.action_norm          = nn.LayerNorm(self.feature_size, elementwise_affine=True)
        self.dropout_0            = nn.Dropout(p=self.drop_rate)

        neural_types = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }
        self.bidirectional        = False
        self.recurrent_layers     = neural_types[self.neural_type.lower()](self.feature_size, self.feature_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=self.bidirectional)

        self.dropout_1            = nn.Dropout(p=self.drop_rate)
        self.reward_linear        = nn.Linear(self.feature_size, self.reward_size, bias=self.bias)
        
        self.state_bias           = nn.Parameter(torch.zeros(self.feature_size) - 1.5)
        self.action_bias          = nn.Parameter(torch.zeros(self.feature_size) + 1.5)

        # Initialize weights for fully connected layers
        self.initialize_weights(self.init  )

        # Optimizer
        optimizers = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adamw': optim.AdamW
        }
        self.selected_optimizer = optimizers[self.opti.lower()](self.parameters(), lr=self.alpha, weight_decay=self.L2_lambda)

        # Loss function
        losses = {
            'mean_squared_error': torch.nn.MSELoss(reduction='mean'),
            'binary_crossentropy': torch.nn.BCELoss(reduction='mean'),
            'huber_loss': torch.nn.SmoothL1Loss(reduction='mean')
        }
        self.loss_function = losses[self.loss .lower()]

        # Loss function
        losses = {
            'mean_squared_error': torch.nn.MSELoss(reduction='none'),
            'binary_crossentropy': torch.nn.BCELoss(reduction='none'),
            'huber_loss': torch.nn.SmoothL1Loss(reduction='none')
        }
        self.loss_function_ = losses[self.loss .lower()]




    def forward(self, history_s, present_s, future_a):

        if history_s.size(1) > 0:
            history_s = self.state_norm(self.state_linear (history_s              ))
            present_s = self.state_norm(self.state_linear (present_s.unsqueeze(1) ))
            future_a  = self.action_norm(self.action_linear(future_a              ))
            history_s = F.gelu(history_s) + self.state_bias 
            present_s = F.gelu(present_s) + self.state_bias 
            future_a  = F.gelu(future_a ) + self.action_bias
            h = torch.cat([history_s, present_s, future_a], dim=1)
        else:
            present_s = self.state_norm(self.state_linear (present_s.unsqueeze(1)))
            future_a  = self.action_norm(self.action_linear(future_a             ))
            present_s = F.gelu(present_s) + self.state_bias 
            future_a  = F.gelu(future_a ) + self.action_bias
            h = torch.cat([present_s, future_a], dim=1)

        h = self.dropout_0(h)
        
        """
        Transformer decoder
        """
        h, _ = self.recurrent_layers(h)
        """
        Transformer decoder
        """

        h = self.dropout_1(h)
        r = self.reward_linear(h)
        r = torch.tanh(r)  

        future_r = r[:, -future_a.size(1): , :]

        return future_r
    



    def initialize_weights(self, initializer):
        initializers = {
            'random_uniform': nn.init.uniform_,
            'random_normal': nn.init.normal_,
            'glorot_uniform': nn.init.xavier_uniform_,
            'glorot_normal': nn.init.xavier_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_, # since we are using nn.linear -> norm layer -> gelu , we don't really need kaiming for gelu
            'kaiming_normal': nn.init.kaiming_normal_
        }
        initializer = initializers[initializer.lower()]
        for module in self.modules():
            if isinstance(module, nn.Linear):
                initializer(module.weight)     # module.weight and module.bias are parameters
                if module.bias is not None:   
                    nn.init.zeros_(module.bias)









