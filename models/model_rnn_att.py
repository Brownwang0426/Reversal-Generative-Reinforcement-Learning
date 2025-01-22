
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
    def __init__(self, feature_size, num_heads):
        super(custom_attn, self).__init__()
        assert feature_size % num_heads == 0, "feature_size must be divisible by num_heads"
        self.bias          = False
        self.feature_size  = feature_size
        self.num_heads     = num_heads
        self.head_size     = feature_size // num_heads
        self.W_q           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_k           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_v           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_o           = nn.Linear(feature_size, feature_size, bias=self.bias)

    def split_heads(self, x):
        batch_size, sequence_size, feature_size = x.size()
        return x.view(batch_size, sequence_size, self.num_heads, self.head_size).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        mask_1, mask_2 = mask
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5) #  (batch_size, num_heads, sequence_size, head_size) @ (batch_size, num_heads, head_size, sequence_size ) 
        
        if mask_1 != None:
            attn_scores += mask_1                                # (batch_size, num_heads, sequence_size, sequence_size) += (batch_size, 1, sequence_size, sequence_size)
        else:
            attn_scores += 0

        if mask_2 != None:
            attn_probs = torch.softmax(attn_scores, dim=-1) * mask_2 # (batch_size, num_heads, sequence_size, sequence_size) * (batch_size, 1, sequence_size, sequence_size)
        else:
            attn_probs = torch.softmax(attn_scores, dim=-1) 

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

        self.state_linear         = nn.Linear(self.h_input_neuron_size, self.hidden_neuron_size, bias=self.bias)
        self.action_linear        = nn.Linear(self.input_neuron_size  , self.hidden_neuron_size, bias=self.bias)
        self.positional_encoding  = nn.Parameter(self.generate_positional_encoding(2 * self.input_sequence_size, self.hidden_neuron_size ), requires_grad=False)
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
        self.reward_linear        = nn.Linear(2 * self.input_sequence_size * self.hidden_neuron_size, self.output_neuron_size , bias=self.bias)
        self.state_linear_        = nn.Linear(2 * self.input_sequence_size * self.hidden_neuron_size, self.h_input_neuron_size, bias=self.bias)

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


    def forward(self, history_s_list, history_a_list, s, a_list):
        
        device    = next(self.parameters()).device 

        if len(history_a_list) > 0:

            r_list = list()
            s_list = list()

            stack_list = list()

            for i in range(history_s_list.size(1)):
                history_s  = self.state_linear(history_s_list[:, i].unsqueeze(1))
                history_s  = self.hidden_activation(history_s)
                stack_list.append(history_s)
                history_a  = self.action_linear(history_a_list[:,i].unsqueeze(1))
                history_a  = self.hidden_activation(history_a)
                stack_list.append(history_a)

            s  = self.state_linear(s.unsqueeze(1))
            s  = self.hidden_activation(s)
            stack_list.append(s)

            for i in range(a_list.size(1)):

                a  = self.action_linear(a_list[:,i].unsqueeze(1))
                a  = self.hidden_activation(a)
                stack_list.append(a)

                h  = torch.cat(stack_list, dim=1)
                h  = h + self.positional_encoding[:, :h.size(1), :]

                original_size = h.size(1)
                pad_size      = 2 * self.input_sequence_size - original_size

                h             = F.pad(h, pad=(0, 0, 0, pad_size), mode='constant', value= 0) 
    
                mask_1        = torch.zeros(original_size, original_size).float().to(device)
                mask_1        = F.pad(mask_1, pad=(0, pad_size, 0, pad_size), mode='constant', value= -1e20) 
    
                mask_2        = torch.ones(original_size, original_size).float().to(device)
                mask_2        = F.pad(mask_2, pad=(0, pad_size, 0, pad_size), mode='constant', value= 0) 
    
                mask          = (mask_1, mask_2)




                h_list = list()
                for j, layer in enumerate(self.transformer_layers):
                    attention_layer, attention_norm_layer, fully_connected_layer, fully_connected_norm_layer = layer
                    if i == 0:
                        h_ = attention_layer(h, h, h, mask)
                    else:
                        h_ = attention_layer(prev_h_list[j], prev_h_list[j], h, mask)
                    h  = attention_norm_layer(h + h_)
                    h_ = fully_connected_layer(h)
                    h  = fully_connected_norm_layer(h + h_)
                    h_list.append(h)
                prev_h_list = h_list




                h  = h.view(h.size(0), -1)

                r  = self.reward_linear(h)   
                r  = self.output_activation(r)

                s  = self.state_linear_(h)   
                s  = self.hidden_activation(s)

                r_list.append(r)
                s_list.append(s)

                s  = self.state_linear(s.unsqueeze(1))
                s  = self.hidden_activation(s)
                stack_list.append(s)

            r_list = torch.stack(r_list, dim=0) # r_list becomes [sequence_size, batch_size, feature_size]
            s_list = torch.stack(s_list, dim=0) # s_list becomes [sequence_size, batch_size, feature_size]
            r_list = r_list.permute(1, 0, 2)    # r_list becomes [batch_size, sequence_size, feature_size]
            s_list = s_list.permute(1, 0, 2)    # s_list becomes [batch_size, sequence_size, feature_size]
        
        else:

            r_list = list()
            s_list = list()

            stack_list = list()

            s  = self.state_linear(s.unsqueeze(1))
            s  = self.hidden_activation(s)
            stack_list.append(s)

            for i in range(a_list.size(1)):

                a  = self.action_linear(a_list[:,i].unsqueeze(1))
                a  = self.hidden_activation(a)
                stack_list.append(a)

                h  = torch.cat(stack_list, dim=1)
                h  = h + self.positional_encoding[:, :h.size(1), :]

                original_size = h.size(1)
                pad_size      = 2 * self.input_sequence_size - original_size

                h             = F.pad(h, pad=(0, 0, 0, pad_size), mode='constant', value= 0) 
    
                mask_1        = torch.zeros(original_size, original_size).float().to(device)
                mask_1        = F.pad(mask_1, pad=(0, pad_size, 0, pad_size), mode='constant', value= -1e20) 
    
                mask_2        = torch.ones(original_size, original_size).float().to(device)
                mask_2        = F.pad(mask_2, pad=(0, pad_size, 0, pad_size), mode='constant', value= 0) 
    
                mask          = (mask_1, mask_2)




                h_list = list()
                for j, layer in enumerate(self.transformer_layers):
                    attention_layer, attention_norm_layer, fully_connected_layer, fully_connected_norm_layer = layer
                    if i == 0:
                        h_ = attention_layer(h, h, h, mask)
                    else:
                        h_ = attention_layer(prev_h_list[j], prev_h_list[j], h, mask)
                    h  = attention_norm_layer(h + h_)
                    h_ = fully_connected_layer(h)
                    h  = fully_connected_norm_layer(h + h_)
                    h_list.append(h)
                prev_h_list = h_list




                h  = h.view(h.size(0), -1)

                r  = self.reward_linear(h)   
                r  = self.output_activation(r)

                s  = self.state_linear_(h)   
                s  = self.hidden_activation(s)

                r_list.append(r)
                s_list.append(s)

                s  = self.state_linear(s.unsqueeze(1))
                s  = self.hidden_activation(s)
                stack_list.append(s)

            r_list = torch.stack(r_list, dim=0) # r_list becomes [sequence_size, batch_size, feature_size]
            s_list = torch.stack(s_list, dim=0) # s_list becomes [sequence_size, batch_size, feature_size]
            r_list = r_list.permute(1, 0, 2)    # r_list becomes [batch_size, sequence_size, feature_size]
            s_list = s_list.permute(1, 0, 2)    # s_list becomes [batch_size, sequence_size, feature_size]
            
        return r_list, s_list


    def generate_positional_encoding(self, sequence_size, feature_size):
        pe = torch.zeros(sequence_size,feature_size)
        for pos in range(sequence_size):
            for i in range(0,feature_size,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/feature_size)))
                if i + 1 < feature_size:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/feature_size)))
        return pe.unsqueeze(0)  # Shape: (1, sequence_size, feature_size)

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
