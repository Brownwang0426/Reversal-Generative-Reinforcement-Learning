
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




class rms_norm(nn.Module):
    def __init__(self, dim, elementwise_affine=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        if self.weight is not None:
            x = x * self.weight
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
                 num_experts,
                 moe_top_k,
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
        self.num_experts          = num_experts
        self.moe_top_k            = moe_top_k
        self.init                 = init
        self.opti                 = opti
        self.loss                 = loss
        self.bias                 = bias
        self.drop_rate            = drop_rate
        self.alpha                = alpha
        self.L2_lambda            = L2_lambda
        self.grad_clip_value      = grad_clip_value

        self.state_linear         = nn.Sequential(
                                        nn.Linear(self.state_size, self.feature_size, bias=self.bias),
                                        nn.GELU(),
                                        nn.Linear(self.feature_size, self.feature_size, bias=self.bias)
                                    )
        self.action_linear        = nn.Sequential(
                                        nn.Linear(self.action_size, self.feature_size, bias=self.bias),
                                        nn.GELU(),
                                        nn.Linear(self.feature_size, self.feature_size, bias=self.bias)
                                    )
        self.state_norm           = rms_norm(self.feature_size, elementwise_affine=True)
        self.action_norm          = rms_norm(self.feature_size, elementwise_affine=True)

        self.state_pad            = nn.Parameter(torch.zeros(1, 1, self.feature_size))

        neural_types = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }
        self.bidirectional        = False
        self.recurrent_layers_h   = neural_types[self.neural_type.lower()](self.feature_size, self.feature_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=self.bidirectional)
        self.recurrent_layers_f   = neural_types[self.neural_type.lower()](self.feature_size, self.feature_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=self.bidirectional)

        self.reward_linear        = nn.Sequential(
                                        nn.Linear(self.feature_size, self.reward_size, bias=self.bias)
                                    )

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




    def forward(self, history_s, history_a, present_s, future_s, future_a, pos_skip):

        history = self.state_norm (self.state_linear  (history_s             )) + self.action_norm(self.action_linear(history_a         ))
        present = self.state_norm (self.state_linear  (present_s.unsqueeze(1))) + self.action_norm(self.action_linear(future_a[:, :1, :]))
        future  = self.action_norm(self.action_linear (future_a[:, 1:, :]))
        future  = self.state_pad.expand(future.size(0), future.size(1), -1) + future
        h       = torch.cat([history, present, future], dim=1)

        """
        Transformer decoder
        """
        h, _ = self.recurrent_layers(h)
        """
        Transformer decoder
        """

        h  = h[:, -self.future_size:, :]
        r  = self.reward_linear(h)
        r  = torch.tanh(r)  

        future_r = r
        future_s = torch.zeros((future_a.size(0), future_a.size(1), self.state_size), device=future_a.device, dtype=future_a.dtype)

        return future_r, future_s




    # def _forward(self, history_s, history_a, present_s, future_s, future_a):
    #     return self.forward(history_s, history_a, present_s, future_s, future_a)

    


    def forward_(self, history_s, history_a, present_s, future_s, future_a):
        return self.forward(history_s, history_a, present_s, future_s, future_a, None)




    def initialize_weights(self, initializer):
        initializers = {
            'random_uniform': nn.init.uniform_,
            'random_normal': nn.init.normal_,
            'glorot_uniform': nn.init.xavier_uniform_,
            'glorot_normal': nn.init.xavier_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_, 
            'kaiming_normal': nn.init.kaiming_normal_
        }
        initializer = initializers[initializer.lower()]
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                initializer(module.weight)     # module.weight and module.bias are parameters
                if module.bias is not None:   
                    nn.init.zeros_(module.bias)
                # if "reward_linear" in name:
                #     if module.bias is not None:
                #         nn.init.constant_(module.bias, 2.0)  # ★ key to make agent optimisitc and explore
                # else:
                #     initializer(module.weight)     # module.weight and module.bias are parameters
                #     if module.bias is not None:   
                #         nn.init.zeros_(module.bias)