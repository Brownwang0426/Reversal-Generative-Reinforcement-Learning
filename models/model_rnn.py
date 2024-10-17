
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
        self.recurrent_layer        = neural_types[self.neural_type.lower()](self.input_neuron_size, self.h_input_neuron_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate)
        self.reward_linear          = nn.Linear(self.h_input_neuron_size, self.output_neuron_size, bias=self.bias)
        self.state_linear           = nn.Linear(self.h_input_neuron_size, self.h_input_neuron_size, bias=self.bias) 

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

        idx = 1 # the index of the num_layers where you want to insert s

        # s      is [batch_size, feature_size] by default
        # a_list is [batch_size, sequence_size, feature_size] by default

        r_list = list()
        s_list = list()

        if self.neural_type == 'lstm':
            cn      = torch.zeros_like(s).repeat(self.num_layers, 1, 1) 
            sn      = torch.zeros_like(s).repeat(self.num_layers, 1, 1) # sn is [num_layers, batch_size, feature_size]
            sn[idx] = s
            rl, scn = self.recurrent_layer(a_list[:, 0, :].unsqueeze(1), (sn, cn)) # a_list[:, 0, :] is [batch_size, sequence_size=0, feature_size]
            r       = rl[:,0,:]  # rl[:,0,:] is [batch_size, sequence_size=0, feature_size] 
            cn      = scn[1]
            c       = cn[idx]
            sn      = scn[0]     # sn[0]     is [tuple_size=0, num_layers, batch_size, feature_size]
            s       = sn[idx]
        else:
            sn      = torch.zeros_like(s).repeat(self.num_layers, 1, 1) # sn is [num_layers, batch_size, feature_size]
            sn[idx] = s
            rl, sn  = self.recurrent_layer(a_list[:, 0, :].unsqueeze(1), sn)        # a_list[:, 0, :] is [batch_size, sequence_size=0, feature_size]
            r       = rl[:,0,:]  # rl[:,0,:] is [batch_size, sequence_size=0, feature_size] 
            sn      = sn         # sn        is [num_layers, batch_size, feature_size]
            s       = sn[idx]

        r  = self.reward_linear(r)    
        r  = self.output_activation(r)

        s  = self.state_linear(s) 
        s  = self.output_activation(s)

        r_list.append(r)  # r_list is [sequence_size, batch_size, feature_size]
        s_list.append(s)  # s_list is [sequence_size, batch_size, feature_size]

        for i in range(a_list.size(1)-1):

            if self.neural_type == 'lstm':
                cn      = torch.zeros_like(s).repeat(self.num_layers, 1, 1) 
                cn[idx] = c
                sn      = torch.zeros_like(s).repeat(self.num_layers, 1, 1) 
                sn[idx] = s
                rl, scn = self.recurrent_layer(a_list[:, i+1, :].unsqueeze(1), (sn, cn))
                r       = rl[:,0,:]
                cn      = scn[1]
                c       = cn[idx]
                sn      = scn[0]
                s       = sn[idx]
            else:
                sn      = torch.zeros_like(s).repeat(self.num_layers, 1, 1) 
                sn[idx] = s
                rl, sn  = self.recurrent_layer(a_list[:, i+1, :].unsqueeze(1), sn)
                r       = rl[:,0,:]
                sn      = sn
                s       = sn[idx]

            r  = self.reward_linear(r)   
            r  = self.output_activation(r)

            s  = self.state_linear(s) 
            s  = self.output_activation(s)

            r_list.append(r)
            s_list.append(s) 

        r_list = torch.stack(r_list, dim=1) # r_list becomes [batch_size, sequence_size, feature_size]
        s_list = torch.stack(s_list, dim=1) # s_list becomes [batch_size, sequence_size, feature_size]

        return r_list, s_list


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