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
                 input_neuron_size_,
                 input_neuron_size,
                 output_neuron_size,
                 hidden_neuron_size,
                 input_sequence_size,
                 neural_type,
                 num_layers,
                 num_heads,
                 initializer,
                 optimizer,
                 loss,
                 bias,
                 drop_rate,
                 alpha):

        super(build_model, self).__init__()

        self.input_neuron_size_   = input_neuron_size_
        self.input_neuron_size    = input_neuron_size
        self.output_neuron_size   = output_neuron_size
        self.hidden_neuron_size   = hidden_neuron_size
        self.input_sequence_size  = input_sequence_size
        self.neural_type          = neural_type
        self.num_layers           = num_layers
        self.num_heads            = num_heads
        self.initializer          = initializer
        self.optimizer            = optimizer
        self.loss                 = loss
        self.bias                 = bias
        self.drop_rate            = drop_rate
        self.alpha                = alpha

        self.state_linear         = nn.Linear(self.input_neuron_size_ , self.hidden_neuron_size, bias=self.bias)
        self.action_linear        = nn.Linear(self.input_neuron_size  , self.hidden_neuron_size, bias=self.bias)
        neural_types = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }
        bidirectional             = False
        self.recurrent_layer      = neural_types[self.neural_type.lower()](self.hidden_neuron_size, self.hidden_neuron_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate, bidirectional=bidirectional)
        self.reward_linear        = nn.Linear(self.hidden_neuron_size, self.output_neuron_size , bias=self.bias)
        self.state_linear_        = nn.Linear(self.hidden_neuron_size, self.input_neuron_size_ , bias=self.bias)
        self.norm_layer_          = nn.LayerNorm(self.input_neuron_size_, elementwise_affine=True) 

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

        r_list = list()
        s_list = list()

        stack_list = list()
        for i in range(history_s_list.size(1)):
            history_s  = self.state_linear(history_s_list[:, i].unsqueeze(1))
            stack_list.append(history_s)
            history_a  = self.action_linear(history_a_list[:,i].unsqueeze(1))
            stack_list.append(history_a)
        s  = self.state_linear(s.unsqueeze(1))
        stack_list.append(s)

        for i in range(a_list.size(1)):

            a  = self.action_linear(a_list[:,i].unsqueeze(1))
            stack_list.append(a)

            h    = torch.cat(stack_list, dim=1)

            """
            RNN, GRU, LSTM
            """
            h, _ = self.recurrent_layer(h)
            
            """
            We utilize the last idx in h to derive the latest reward and state.
            """
            r  = self.reward_linear(h[:, - 1, :])   
            r  = torch.tanh(r)
            s  = self.state_linear_(h[:, - 1, :])   
            s  = self.norm_layer_(s)

            r_list.append(r)
            s_list.append(s)

            """
            We save the latest state into the next round or time step.
            """
            s  = self.state_linear(s.unsqueeze(1))
            stack_list.append(s)

        r_list = torch.stack(r_list, dim=0) # r_list becomes [sequence_size, batch_size, feature_size]
        s_list = torch.stack(s_list, dim=0) # s_list becomes [sequence_size, batch_size, feature_size]
        r_list = r_list.permute(1, 0, 2)    # r_list becomes [batch_size, sequence_size, feature_size]
        s_list = s_list.permute(1, 0, 2)    # s_list becomes [batch_size, sequence_size, feature_size]
    
        return r_list, s_list
    



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
