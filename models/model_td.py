
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
        self.attn_dropout  = DeterministicDropout(self.drop_rate)
        self.resid_dropout = DeterministicDropout(self.drop_rate)

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
        self.dropout_0            = DeterministicDropout(self.drop_rate)

        self.positional_encoding  = nn.Parameter(self.generate_positional_encoding(self.history_size + 1 + self.future_size , self.feature_size ), requires_grad=False)
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
        mask                      = torch.full((1, 1, self.history_size + 1 + self.future_size, self.history_size + 1 + self.future_size), float("-inf"))
        mask                      = torch.triu(mask , diagonal=1)
        self.register_buffer('mask', mask)  

        self.dropout_1            = DeterministicDropout(self.drop_rate)
        self.reward_linear        = nn.Linear(self.feature_size, self.reward_size  , bias=self.bias)

        self.state_bias           = nn.Parameter(torch.zeros(self.feature_size) - 1.5)
        self.action_bias          = nn.Parameter(torch.zeros(self.feature_size) + 1.5)

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




    def forward(self, history_s, present_s, future_a):

        if history_s.size(1) > 0:
            history_s = self.state_linear (history_s)  
            present_s = self.state_linear (present_s.unsqueeze(1))
            future_a  = self.action_linear(future_a) 
            history_s = torch.tanh(history_s) + self.state_bias
            present_s = torch.tanh(present_s) + self.state_bias
            future_a  = torch.tanh(future_a ) + self.action_bias
            h = torch.cat([history_s, present_s, future_a], dim=1)
        else:
            present_s = self.state_linear (present_s.unsqueeze(1))
            future_a  = self.action_linear(future_a) 
            present_s = torch.tanh(present_s) + self.state_bias
            future_a  = torch.tanh(future_a ) + self.action_bias
            h = torch.cat([present_s, future_a], dim=1)

        h = self.dropout_0(h)

        """
        Transformer decoder
        """
        long = h.size(1)
        h = h + self.positional_encoding[:, :long, :]
        for layer in self.transformer_layers:
            attention_norm, attention_linear, fully_connected_norm, fully_connected_linear = layer
            h_ = attention_norm(h)
            h  = h + attention_linear(h_, h_, h_, self.mask[:, :, :long, :long])
            h_ = fully_connected_norm(h)
            h  = h + fully_connected_linear(h_)
        h = self.transformer_norm(h)

        """
        We utilize the last idx in h to derive the latest reward and state.
        """
        h = self.dropout_1(h)
        r = self.reward_linear(h[:, -future_a.size(1): , :])  
        future_r = torch.tanh(r)

        return future_r




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
