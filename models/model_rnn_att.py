
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
        # mask -> (batch_size, 1, seq_length, d_model)
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
        self.drop_rate            = drop_rate
        self.alpha                = alpha

        self.bias                 = False

        self.state_linear         = nn.Linear(self.h_input_neuron_size, self.hidden_neuron_size, bias=self.bias)
        self.action_linear        = nn.Linear(self.input_neuron_size  , self.hidden_neuron_size, bias=self.bias)
        self.positional_encoding  = nn.Parameter(self.generate_positional_encoding(2, self.hidden_neuron_size ), requires_grad=False)
        self.transformer_layers   = \
        nn.ModuleList([
            nn.ModuleList([
                custom_attn(self.hidden_neuron_size, self.num_heads),
                nn.LayerNorm(self.hidden_neuron_size),
                nn.Linear(self.hidden_neuron_size, self.hidden_neuron_size, bias=self.bias),
                nn.LayerNorm(self.hidden_neuron_size)
            ])
            for _ in range(self.num_layers)
        ])
        self.reward_linear        = nn.Linear(self.hidden_neuron_size, self.output_neuron_size , bias=self.bias)
        self.state_linear_        = nn.Linear(self.hidden_neuron_size, self.h_input_neuron_size, bias=self.bias)

        # Activation functions
        self.hidden_activation = self.get_activation(self.hidden_activation)
        self.output_activation = self.get_activation(self.output_activation)

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


    def forward(self, s, a_list):
        
        mask = None

        r_list = list()
        s_list = list()

        s  = self.state_linear(s)
        s  = self.hidden_activation(s)

        a_ = self.action_linear(a_list[:,0])
        a_ = self.hidden_activation(a_)

        h  = torch.stack([s, a_], dim=0).view(a_.size(0), 2, a_.size(1))
        # since each step contains only two inputs - present state and present action, we will save positional encoding.
        # h  = h + self.positional_encoding[:, :, :] 

        pres_h_list = list()
        for j, layer in enumerate(self.transformer_layers):
            attention_layer, attention_norm_layer, fully_connected_layer, fully_connected_norm_layer = layer
            h_ = attention_layer(h, h, h, mask)
            h  = attention_norm_layer(h + h_)
            h_ = fully_connected_layer(h)
            h  = fully_connected_norm_layer(h + h_)
            pres_h_list.append(h)
        prev_h_list = pres_h_list

        r  = h[:, 0]
        s_ = h[:, 1]
        
        r  = self.reward_linear(r)   
        r  = self.output_activation(r)

        s_ = self.state_linear_(s_)   
        s_ = self.output_activation(s_)

        s  = self.state_linear(s_)
        s  = self.hidden_activation(s)

        r_list.append(r)
        s_list.append(s_)

        for i in range(a_list.size(1)-1):

            a_ = self.action_linear(a_list[:,i+1])
            a_ = self.hidden_activation(a_)

            h  = torch.stack([s, a_], dim=0).view(a_.size(0), 2, a_.size(1))
            # Since each step contains only two inputs - present state and present action, we will save positional encoding.
            # h  = h + self.positional_encoding[:, :, :]

            pres_h_list = list()
            for j, layer in enumerate(self.transformer_layers):
                attention_layer, attention_norm_layer, fully_connected_layer, fully_connected_norm_layer = layer
                # We decide not to use attention_layer(prev_h_list[j], prev_h_list[j], h, mask) for the following reasons:
                #  1. Since it is not easy to track down prev_h_list[j] for the first step, we find it hard to explain why the following steps can use prev_h_list[j] from the previous step
                #  2. Since the we use attention_layer(h, h, h, mask) for the first step, we should keep using the same method thereafter.
                h_ = attention_layer(h, h, h, mask)
                # h_ = attention_layer(prev_h_list[j], prev_h_list[j], h, mask)
                h  = attention_norm_layer(h + h_)
                h_ = fully_connected_layer(h)
                h  = fully_connected_norm_layer(h + h_)
                pres_h_list.append(h)
            prev_h_list = pres_h_list

            r  = h[:, 0]
            s_ = h[:, 1]
            
            r  = self.reward_linear(r)   
            r  = self.output_activation(r)

            s_ = self.state_linear_(s_)   
            s_ = self.output_activation(s_)

            s  = self.state_linear(s_)
            s  = self.hidden_activation(s)

            r_list.append(r)
            s_list.append(s_)

        r_list = torch.stack(r_list, dim=1)
        s_list = torch.stack(s_list, dim=1)

        return r_list, s_list


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