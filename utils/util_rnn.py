
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




def update_pre_activated_actions(iteration_for_deducing,
                                 model_list,
                                 state,
                                 pre_activated_actions,
                                 desired_reward,
                                 beta,
                                 device):

    state, pre_activated_actions, desired_reward = state.to(device), pre_activated_actions.to(device), desired_reward.to(device)

    model_list_copy = copy.deepcopy(model_list)

    for _ in range(iteration_for_deducing):

        model   = random.choice(model_list_copy)

        actions = torch.tanh(pre_activated_actions)

        model.train()
        actions = actions.clone().detach().requires_grad_(True)
        if actions.grad is not None:
            actions.grad.zero_()
        for param in model.parameters():
            param.requires_grad = False

        loss_function = model.loss_function
        output, _     = model(state, actions, padding_mask=None)
        total_loss    = loss_function(output, desired_reward)
        total_loss.backward() # get grad

        pre_activated_actions -= actions.grad * (1 - actions ** 2) * beta # update params

    return pre_activated_actions




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
    
    for state, actions, reward, next_state, padding_mask in sub_data_loader:

        next_state  = torch.unsqueeze(next_state, dim=0).repeat(model.num_layers, 1, 1)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        output, output_state        = model(state, actions, padding_mask)
        total_loss                  = loss_function(output, reward) + loss_function(output_state, next_state)
        total_loss                 += EWC_loss(EWC_lambda, model, prev_model, prev_gradient_matrix)
        total_loss.backward()     # get grad

        selected_optimizer.step() # update params

    return model




def update_gradient_matrix(model,
                           data_loader):
    
    gradient_matrix = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for state, actions, reward, next_state, padding_mask in data_loader:

        next_state  = torch.unsqueeze(next_state, dim=0).repeat(model.num_layers, 1, 1)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function        = model.loss_function
        output, output_state = model(state, actions, padding_mask)
        total_loss           = loss_function(output, reward) + loss_function(output_state, next_state)
        total_loss.backward()        # get grad

        for name, param in model.named_parameters():
            gradient_matrix[name] += param.grad

    gradient_matrix = {name: param / len(data_loader) for name, param in gradient_matrix.items()}

    return gradient_matrix




def initialize_pre_activated_actions(init, noise_t, noise_r, shape):
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




def sequentialize(state_list, action_list, reward_list, time_size):

    sequentialized_state_list       = []
    sequentialized_action_list      = []
    sequentialized_reward_list      = []
    sequentialized_next_state_list  = []

    if time_size > len(state_list[:-1]):
        time_size = len(state_list[:-1])
    else:
      pass

    # time_size_ = time_size
    # for i in range(len(reward_list[:])):
    #     sequentialized_state_list.append(       state_list [ i ] )
    #     sequentialized_action_list.append(      action_list[ i : i+time_size_]  )
    #     sequentialized_reward_list.append(      reward_list[ i + len(action_list[i:i+time_size_]) - 1 ]  )
    #     sequentialized_next_state_list.append(  state_list [ i + len(action_list[i:i+time_size_])     ]  )

    # a more sophisticated method
    for j in range(time_size):
        time_size_ = j+1
        if time_size_== 1:
            for i in range(len(reward_list[:])):
                sequentialized_state_list.append(       state_list [ i ] )
                sequentialized_action_list.append(      action_list[ i : i+time_size_]  )
                sequentialized_reward_list.append(      reward_list[ i + len(action_list[i:i+time_size_]) - 1 ]  )
                sequentialized_next_state_list.append(  state_list [ i + len(action_list[i:i+time_size_])     ]  )
        else:
            for i in range(len(reward_list[:-time_size_+1])):
                sequentialized_state_list.append(       state_list [ i ] )
                sequentialized_action_list.append(      action_list[ i : i+time_size_]  )
                sequentialized_reward_list.append(      reward_list[ i + len(action_list[i:i+time_size_]) - 1 ]  )
                sequentialized_next_state_list.append(  state_list [ i + len(action_list[i:i+time_size_])     ]  )

    return sequentialized_state_list, sequentialized_action_list, sequentialized_reward_list, sequentialized_next_state_list




def obtain_tensor_from_list(sequentialized_state_list,
                            sequentialized_actions_list,
                            sequentialized_reward_list,
                            sequentialized_next_state_list,
                            time_size,
                            mask_value,
                            num_heads,
                            device):

    # Convert lists to tensors directly on the desired device and data type
    state_tensor = torch.tensor(np.array(sequentialized_state_list), dtype=torch.float).to(device)
    reward_tensor = torch.tensor(np.array(sequentialized_reward_list), dtype=torch.float).to(device)
    next_state_tensor = torch.tensor(np.array(sequentialized_next_state_list), dtype=torch.float).to(device)

    # Pad and stack actions_tensor efficiently
    actions_list = []
    for arr in sequentialized_actions_list:
        tensor_arr = torch.tensor(np.array(arr), dtype=torch.float).to(device)
        # Pad tensor only once per tensor
        if tensor_arr.size(0) < time_size:
            padded_arr = F.pad(tensor_arr,
                               (0, 0, 0, time_size - tensor_arr.size(0)),
                               mode='constant',
                               value=mask_value)
        else:
            padded_arr = tensor_arr
        actions_list.append(padded_arr)
    actions_tensor = torch.stack(actions_list).to(device)

    # Compute row_mask and padding_mask efficiently
    row_mask = (actions_tensor == mask_value).all(dim=-1)
    padding_mask = row_mask.to(dtype=torch.bool)
    padding_mask = padding_mask.to(device)

    return state_tensor, actions_tensor, reward_tensor, next_state_tensor, padding_mask




def obtain_TD_error(model,
                    data_loader):

    for state, actions, reward, next_state, padding_mask in data_loader:

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function        = model.loss_function_
        output, _            = model(state, actions, padding_mask)
        total_loss           = loss_function(output, reward).detach()
        total_loss           = torch.sum(torch.abs(total_loss), axis=1)

    return total_loss



def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)

        