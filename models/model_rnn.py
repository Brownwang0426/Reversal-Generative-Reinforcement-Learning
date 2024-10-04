
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

        self.bias = False

        neural_types = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }
        self.recurrent_layer        = neural_types[self.neural_type.lower()](self.input_neuron_size, self.h_input_neuron_size, num_layers=self.num_layers, batch_first=True, bias=self.bias, dropout=self.drop_rate)
        self.reward_linear          = nn.Linear(self.h_input_neuron_size, self.output_neuron_size, bias=self.bias)
        self.state_linear_          = nn.Linear(self.h_input_neuron_size, self.h_input_neuron_size, bias=self.bias)

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

        r_list = list()
        s_list = list()

        if self.neural_type == 'lstm':
            s       = torch.unsqueeze(s, dim=0).repeat(self.num_layers, 1, 1)
            r, s_   = self.recurrent_layer(a_list[:, 0, :].unsqueeze(1), (s, s))
            r       = r[0]
            s_      = s_[0]
        else:
            s       = torch.unsqueeze(s, dim=0).repeat(self.num_layers, 1, 1)
            r, s_   = self.recurrent_layer(a_list[:, 0, :].unsqueeze(1), s)
            r       = r[0]
            
        r  = self.reward_linear(r)   
        r  = self.output_activation(r)

        s_ = self.state_linear_(s_)   
        s_ = self.output_activation(s_)

        r_list.append(r)
        s_list.append(s_)

        for i in range(a_list.size(1)-1):

            if self.neural_type == 'lstm':
                r, s_   = self.recurrent_layer(a_list[:, i+1, :].unsqueeze(1), (s_, s_))
                r       = r[0]
                s_      = s_[0]
            else:
                r, s_   = self.recurrent_layer(a_list[:, i+1, :].unsqueeze(1), s_)
                r       = r[0]

            r  = self.reward_linear(r)   
            r  = self.output_activation(r)

            s_ = self.state_linear_(s_)   
            s_ = self.output_activation(s_)

            r_list.append(r)
            s_list.append(s_)

        r_list = torch.stack(r_list, dim=1)
        s_list = torch.stack(s_list, dim=0).permute(1, 2, 0, 3)

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