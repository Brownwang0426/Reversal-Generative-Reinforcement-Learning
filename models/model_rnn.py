
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
        self.state_linears_in       = nn.ModuleList([nn.Linear(self.h_input_neuron_size, self.h_input_neuron_size) for _ in range(self.num_layers)])
        self.state_linears_out      = nn.ModuleList([nn.Linear(self.h_input_neuron_size, self.h_input_neuron_size) for _ in range(self.num_layers)])

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

        # s      is [batch_size, feature_size] by default
        # a_list is [batch_size, sequence_size, feature_size] by default

        r_list = list()
        s_list = list()

        for _ in range(a_list.size(1)):

            if self.neural_type == 'lstm':
                cl      = torch.zeros_like(s).repeat(self.num_layers, 1, 1) 
                sl      = torch.zeros_like(s).repeat(self.num_layers, 1, 1) # sl is [num_layers, batch_size, feature_size]
                sl[idx] = s
                rl, scl = self.recurrent_layer(a_list[:, 0, :].unsqueeze(1), (sl, cl)) # a_list[:, 0, :] is [batch_size, sequence_size=0, feature_size]
                sl      = scl[0]     # sl[0]     is [tuple_size=0, num_layers, batch_size, feature_size]
                cl      = scl[1]
                r       = rl[:,0,:]  # rl[:,0,:] is [batch_size, sequence_size=0, feature_size] 
                s       = sl[idx]
                c       = cl[idx]
            else:
                sl      = [torch.tanh(state_linear(s)) for state_linear in self.state_linears_in]
                sl      = [s.unsqueeze(0) for s in sl] 
                sl      = torch.cat(sl, dim=0)    # sl        is [num_layers, batch_size, feature_size]
                
                r , sl  = self.recurrent_layer(a_list[:, 0, :].unsqueeze(1), sl)        # a_list[:, 0, :] is [batch_size, sequence_size=0, feature_size]
                
                sl      = [state_linear(sl[i]) for i, state_linear in enumerate(self.state_linears_out)]
                sl      = torch.stack(sl, dim=0)
                s       = sl.sum(dim=0)
                s       = torch.tanh(s)

                r       = r[:,0,:]                # rl[:,0,:] is [batch_size, sequence_size=0, feature_size] 
                r       = self.reward_linear(r)    
                r       = self.output_activation(r)

            r_list.append(r)  # r_list is [sequence_size, batch_size, feature_size]
            s_list.append(s)  # s_list is [sequence_size, batch_size, feature_size]

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