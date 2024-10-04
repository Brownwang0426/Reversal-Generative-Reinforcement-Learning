
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



def update_pre_activated_action(epoch_for_deducing,
                                model_list,
                                state,
                                pre_activated_future_action,
                                desired_future_reward,
                                beta,
                                device):
    
    state, pre_activated_future_action, desired_future_reward = state.to(device), pre_activated_future_action.to(device), desired_future_reward.to(device)

    model_list_copy = copy.deepcopy(model_list)

    for epoch in range(epoch_for_deducing):

        random.shuffle(model_list_copy)

        for model in model_list_copy:

            future_action = torch.sigmoid(pre_activated_future_action)

            model.train()
            future_action = future_action.clone().detach().requires_grad_(True)
            if future_action.grad is not None:
                future_action.grad.zero_()
            for param in model.parameters():
                param.requires_grad = False

            loss_function       = model.loss_function
            output_reward, _    = model(state, future_action)
            total_loss          = loss_function(output_reward, desired_future_reward)
            total_loss.backward() # get grad

            pre_activated_future_action -= future_action.grad * (1 - future_action) * future_action * beta # update params

    return pre_activated_future_action




# traditional EWC
def EWC_loss(EWC_lambda, model, prev_model, prev_gradient_matrix):
    model_param      = model.state_dict()
    prev_model_param = prev_model.state_dict()
    loss = 0
    for name, param in model.named_parameters():
        diagonal_fisher_matrix = prev_gradient_matrix[name] ** 2
        param_diff             = (model_param[name] - prev_model_param[name]) ** 2
        loss                  += (diagonal_fisher_matrix * param_diff).sum()
    return EWC_lambda * loss




def update_model(model,
                 sub_data_loader,
                 prev_model,
                 prev_gradient_matrix,
                 EWC_lambda):

    for state, future_action, future_reward, future_state, padding_mask in sub_data_loader:

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        output_reward, output_state = model(state, future_action)
        total_loss                  = loss_function(output_reward, future_reward) + loss_function(output_state, future_state)
        total_loss                 += EWC_loss(EWC_lambda, model, prev_model, prev_gradient_matrix)
        total_loss.backward()     # get grad

        selected_optimizer.step() # update params

    return model




def update_gradient_matrix(model,
                           data_loader):
    
    gradient_matrix = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for state, future_action, future_reward, future_state, padding_mask in data_loader:

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        output_reward, output_state = model(state, future_action)
        total_loss                  = loss_function(output_reward, future_reward) + loss_function(output_state, future_state)
        total_loss.backward()        # get grad

        for name, param in model.named_parameters():
            if name != "positional_encoding":
                gradient_matrix[name] += param.grad

    gradient_matrix = {name: param / len(data_loader) for name, param in gradient_matrix.items()}

    return gradient_matrix




def initialize_pre_activated_action(init, noise_t, noise_r, shape):
    input = 0
    if   init == "random_uniform":
        for _ in range(noise_t):
            input += np.random.uniform(low=0, high=1, size=shape) * noise_r
    elif init == "random_normal":
        for _ in range(noise_t):
            input += np.random.normal(loc=0.0, scale= 1, size= shape ) * noise_r
    elif init == "glorot_uniform":
        for _ in range(noise_t):
            limit = np.sqrt(6 / (shape[1] + shape[1]))
            input += np.random.uniform(low=-limit, high=limit, size=shape) * noise_r
    elif init == "glorot_normal":
        for _ in range(noise_t):
            input += np.random.normal(loc=0.0, scale= np.sqrt(2 / (shape[1] + shape[1])) , size= shape ) * noise_r
    elif init == "xavier_uniform":
        for _ in range(noise_t):
            limit = np.sqrt(6 / (shape[1] + shape[1]))
            input += np.random.uniform(low=-limit, high=limit, size=shape) * noise_r
    elif init == "xavier_normal":
        for _ in range(noise_t):
            input += np.random.normal(loc=0.0, scale= np.sqrt(2 / (shape[1] + shape[1])) , size= shape ) * noise_r
    return input




def sequentialize(short_term_state_list, short_term_action_list, short_term_reward_list, chunk_size):

    short_term_present_state_list = []
    short_term_future_action_list = []
    short_term_future_reward_list = []
    short_term_future_state_list  = []

    if chunk_size > len(short_term_state_list[:-1]):
        chunk_size = len(short_term_state_list[:-1])
    else:
      pass

    for i in range(len(short_term_reward_list[:-chunk_size+1])):
        short_term_present_state_list.append(      short_term_state_list [ i                        ]  )
        short_term_future_action_list.append(      short_term_action_list[ i   : i+chunk_size       ]  )
        short_term_future_reward_list.append(      short_term_reward_list[ i   : i+chunk_size       ]  )
        short_term_future_state_list.append(       short_term_state_list [ i+1 : i+chunk_size+1     ]  )

    return short_term_present_state_list, short_term_future_action_list, short_term_future_reward_list, short_term_future_state_list




def obtain_tensor_from_list(short_term_state_list,
                            short_term_future_action_list,
                            short_term_future_reward_list,
                            short_term_future_state_list,
                            time_size,
                            mask_value,
                            num_heads,
                            device):

    # Convert lists to tensors directly on the desired device and data type
    short_term_state_tensor         = torch.tensor(np.array(short_term_state_list), dtype=torch.float).to(device)
    short_term_future_action_tensor = torch.tensor(np.array(short_term_future_action_list), dtype=torch.float).to(device)
    short_term_future_reward_tensor = torch.tensor(np.array(short_term_future_reward_list), dtype=torch.float).to(device)
    short_term_future_state_tensor  = torch.tensor(np.array(short_term_future_state_list), dtype=torch.float).to(device)
    dummy                           = torch.tensor(np.array(short_term_future_action_list), dtype=torch.float).to(device)

    return short_term_state_tensor, short_term_future_action_tensor, short_term_future_reward_tensor, short_term_future_state_tensor, dummy




def obtain_TD_error(model,
                    data_loader):

    for state, future_action, future_reward, future_state, padding_mask in data_loader:

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function        = model.loss_function_
        output_reward, _     = model(state, future_action)
        total_loss           = loss_function(output_reward, future_reward).detach()
        total_loss           = torch.sum(torch.abs(total_loss), dim=(1, 2))

    return total_loss




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)






