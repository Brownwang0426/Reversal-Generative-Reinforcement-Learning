import gym

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

"""
# Model for agent
Crucial model regarding how you set up your agent's neural network

- We suggest you prerpare enough layers between present state and next state.
- We suggest you prerpare enough layers between state and reward.
- We suggest you prerpare enough layers between state and action.
- In our experience, how the neural net of the agent handles the information flow toward reward will have immense impact on the performance of the agent. 
"""


class custom_attn(nn.Module):
    def __init__(self, d_model, num_heads = 8):
        super(custom_attn, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.bias      = False
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q  = nn.Linear(d_model, d_model, bias=self.bias)
        self.W_k  = nn.Linear(d_model, d_model, bias=self.bias)
        self.W_v  = nn.Linear(d_model, d_model, bias=self.bias)
        self.W_o  = nn.Linear(d_model, d_model, bias=self.bias)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask != None:
            attn_scores += mask
        else:
            attn_scores += 0

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output     = torch.matmul(attn_probs, V)

        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        #  (batch_size, seq_length, d_model) - > (batch_size, seq_length, self.num_heads, self.d_k) -> (batch_size, self.num_heads, seq_length, self.d_k)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Q    -> (batch_size, seq_length, d_model)
        # mask -> (batch_size, 1, seq_length, seq_length)
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output      = self.W_o(self.combine_heads(attn_output))
        return output


class build_model(nn.Module):
    def __init__(self,
                 h_input_neuron_size,
                 hidden_neuron_size,
                 input_neuron_size,
                 input_sequence_size,
                 output_neuron_size,
                 neural_type,
                 num_layers,
                 num_heads,
                 hidden_activation,
                 output_activation,
                 shift,
                 initializer,
                 optimizer,
                 loss,
                 bias,
                 drop_rate,
                 alpha):

        super(build_model, self).__init__()

        self.h_input_neuron_size  = h_input_neuron_size
        self.hidden_neuron_size   = hidden_neuron_size
        self.input_neuron_size    = input_neuron_size
        self.input_sequence_size  = input_sequence_size
        self.output_neuron_size   = output_neuron_size
        self.neural_type          = neural_type
        self.num_layers           = num_layers
        self.num_heads            = num_heads
        self.hidden_activation    = hidden_activation
        self.output_activation    = output_activation
        self.shift                = shift
        self.initializer          = initializer
        self.optimizer            = optimizer
        self.loss                 = loss
        self.bias                 = bias
        self.drop_rate            = drop_rate
        self.alpha                = alpha


        neural_types = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }
        self.recurrent_layer_1    = neural_types[self.neural_type.lower()](self.input_neuron_size, self.h_input_neuron_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate)
        self.recurrent_layer_2    = neural_types[self.neural_type.lower()](self.input_neuron_size, self.h_input_neuron_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate)
        self.recurrent_layer_3    = neural_types[self.neural_type.lower()](self.input_neuron_size, self.h_input_neuron_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate)
        self.positional_encoding  = nn.Parameter(self.generate_positional_encoding(self.input_sequence_size, self.h_input_neuron_size ), requires_grad=False)
        self.custom_attn          = custom_attn (self.h_input_neuron_size, self.num_heads)
        self.norm_layer_1         = nn.LayerNorm(self.h_input_neuron_size)
        self.linear_layer_1       = nn.Linear(self.h_input_neuron_size, self.h_input_neuron_size, bias=self.bias)
        self.norm_layer_2         = nn.LayerNorm(self.h_input_neuron_size)
        self.linear_layer_2       = nn.Linear(self.h_input_neuron_size * self.input_sequence_size, self.output_neuron_size, bias=self.bias)

        # Activation functions
        self.hidden_activation    = self.get_activation(self.hidden_activation)
        self.output_activation    = self.get_activation(self.output_activation)

        # Initialize weights for fully connected layers
        self.initialize_weights(self.initializer  )

        # Optimizer
        optimizers = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        self.selected_optimizer = optimizers[self.optimizer.lower()](self.parameters(), lr=self.alpha)

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


    def forward(self, s, al):

        null_step = torch.zeros_like(al[:, 0, :]).unsqueeze(1)

        idx = 0 # the index of the num_layers where you want to insert s

        # s          is [batch_size, feature_size]
        # al         is [batch_size, sequence_size, feature_size]
        # cl         is [num_layers, batch_size, feature_size]
        # sl         is [num_layers, batch_size, feature_size]
        # rl         is [batch_size, sequence_size, feature_size] 

        r_list = list()
        s_list = list()

        for i in range(al.size(1)):

            if self.neural_type == 'lstm':
                if i == 0:
                    cl       = torch.zeros_like(s).repeat(self.num_layers, 1, 1) - 1
                    sl       = torch.zeros_like(s).repeat(self.num_layers, 1, 1) - 1        
                    sl[idx]  = s  
                else:
                    pass                             
                rl, (sl, cl) = self.recurrent_layer_1(null_step                     , (sl, cl))
                rl, (sl, cl) = self.recurrent_layer_2(al[:, i, :].unsqueeze(1)      , (sl, cl))
                r            = rl[:,0,:] 
                rl, (sl, cl) = self.recurrent_layer_3(null_step                     , (sl, cl))
                s            = sl[idx]
                c            = cl[idx]
            else:
                if i == 0:
                    sl       = torch.zeros_like(s).repeat(self.num_layers, 1, 1) - 1        
                    sl[idx]  = s                                                                         
                else:
                    pass
                rl, sl       = self.recurrent_layer_1(null_step                     , sl)
                rl, sl       = self.recurrent_layer_2(al[:, i, :].unsqueeze(1)      , sl)
                r            = rl[:,0,:]
                rl, sl       = self.recurrent_layer_3(null_step                     , sl)    
                s            = sl[idx]

            r_list.append(r) # r_list is [sequence_size, batch_size, feature_size]
            s_list.append(s) # s_list is [sequence_size, batch_size, feature_size]

        rl = torch.stack(r_list, dim=0) # rl is [sequence_size, batch_size, feature_size]
        sl = torch.stack(s_list, dim=0) # sl is [sequence_size, batch_size, feature_size]
        rl = rl.permute(1, 0, 2)        # rl is [batch_size, sequence_size, feature_size]
        sl = sl.permute(1, 0, 2)        # sl is [batch_size, sequence_size, feature_size]

        ori_size = rl.size(1)
        pad_size = self.input_sequence_size - ori_size
        pad      = torch.zeros(rl.size(0), pad_size, rl.size(2))
        mask     = torch.zeros(rl.size(0), 1, self.input_sequence_size, self.input_sequence_size)
        mask[:, :, ori_size:, :] = float('-inf')
        mask[:, :, :, ori_size:] = float('-inf')

        rl  = torch.cat([rl, pad], dim=1)

        rl  = rl + self.positional_encoding[:, :, :]
        
        rl_ = self.custom_attn   (rl, rl, rl, mask)
        rl  = self.norm_layer_1  (rl + rl_)
        rl_ = self.linear_layer_1(rl)
        rl  = self.norm_layer_2  (rl + rl_)

        r   = torch.flatten(rl, start_dim=1)
        r   = self.linear_layer_2(r)
        r   = self.output_activation(r)

        return r, sl

    def generate_positional_encoding(self, max_len, model_dim):
        pe = torch.zeros(max_len,model_dim)
        for pos in range(max_len):
            for i in range(0,model_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/model_dim)))
                if i + 1 < model_dim:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/model_dim)))
        return pe.unsqueeze(0)  # Shape: (1, max_len, model_dim)

    def get_activation(self,  activation):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        return activations[ activation.lower()]

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
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                initializer(layer.weight)