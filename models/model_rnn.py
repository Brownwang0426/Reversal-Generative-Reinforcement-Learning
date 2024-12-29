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

        self.mask_value = 0
        self.state_layer_in_0      = nn.Linear(self.h_input_neuron_size, self.hidden_neuron_size, bias=self.bias)
        self.state_layer_in_1      = nn.Linear(self.hidden_neuron_size, self.hidden_neuron_size, bias=self.bias)
        self.state_layer_out_0     = nn.Linear(self.hidden_neuron_size, self.hidden_neuron_size, bias=self.bias)
        self.state_layer_out_1     = nn.Linear(self.hidden_neuron_size, self.h_input_neuron_size, bias=self.bias)

        self.recurrent_layer                 = neural_types[neural_type.lower()](self.input_neuron_size, self.hidden_neuron_size, num_layers=self.num_layers, batch_first=False, bias=self.bias)

        self.reward_layer                    = nn.Linear(self.hidden_neuron_size, self.output_neuron_size, bias=self.bias)

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


    def forward(self, s, a):

        s  = self.state_layer_in_0(s)
        s  = self.hidden_activation(s)
        s  = self.state_layer_in_1(s)
        s  = self.hidden_activation(s)

        a          = a.permute(1, 0, 2)
        lengths    = (a != self.mask_value).any(dim=2).sum(dim=0).cpu().long() # since a is (sequence_length, batch_size, input_size), we should use sum(dim=0)
        a          = rnn_utils.pack_padded_sequence(a, lengths, batch_first=False, enforce_sorted=False)

        # Forward propagate RNN
        if self.neural_type == 'lstm':
            cl          = torch.zeros_like(s).repeat(self.num_layers, 1, 1) - 1
            sl          = torch.zeros_like(s).repeat(self.num_layers, 1, 1) - 1        
            sl[0]       = s  
            r, (sl, cl) = self.recurrent_layer(a, (sl, cl))
            s           = sl[0] 
        else:
            sl          = torch.zeros_like(s).repeat(self.num_layers, 1, 1) - 1        
            sl[0]       = s     
            r, sl       = self.recurrent_layer(a, sl)
            s           = sl[0]

        s  = self.state_layer_out_0(s)
        s  = self.hidden_activation(s)
        s  = self.state_layer_out_1(s)
        s  = self.hidden_activation(s)

        r, _  = rnn_utils.pad_packed_sequence(r, batch_first=False)
        r     = r.permute(1, 0, 2)
        r     = r[:, -1]
        r     = self.reward_layer(r)
        r     = self.output_activation(r)

        return r, s


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