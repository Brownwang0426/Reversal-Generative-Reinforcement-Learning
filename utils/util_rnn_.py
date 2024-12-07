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




def update_pre_activated_actions(iteration_for_deducing,
                                model_list,
                                state,
                                pre_activated_future_actions,
                                desired_reward,
                                beta,
                                device):

    state, pre_activated_future_actions, desired_reward = state.to(device), pre_activated_future_actions.to(device), desired_reward.to(device)

    model_list_copy = copy.deepcopy(model_list)

    for i in range(iteration_for_deducing):

        index            = np.random.randint(len(model_list_copy))
        model            = model_list_copy[index]
        
        future_actions   = torch.sigmoid(pre_activated_future_actions)

        model.train()
        future_actions   = future_actions.detach().requires_grad_(True)
        if future_actions.grad is not None:
            future_actions.grad.zero_()
        for param in model.parameters():
            param.requires_grad = False

        loss_function       = model.loss_function
        output_reward, _    = model(state, future_actions)
        total_loss          = loss_function(output_reward, desired_reward)
        total_loss.backward() # get grad

        pre_activated_future_actions -= future_actions.grad * (1 - future_actions) * future_actions * beta # update params
    
    return pre_activated_future_actions




def sequentialize(state_list, action_list, reward_list, chunk_size_, device):

    pad_size = copy.deepcopy(chunk_size_)

    present_state_list  = []
    future_actions_list = []
    future_rewards_list  = []
    future_states_list   = []

    if chunk_size_ > len(state_list[:-1]):
        chunk_size_ = len(state_list[:-1])
    else:
      pass
    
    for j in range(chunk_size_):
        chunk_size = j + 1
        if chunk_size != 1:
            for i in range(len(reward_list[:-chunk_size+1])):
                present_state_list.append(       torch.tensor(np.array(state_list [ i                        ]), dtype=torch.float)  )
                future_actions_list.append(      torch.tensor(np.array(action_list[ i   : i+chunk_size       ]), dtype=torch.float)  )
                future_rewards_list.append(       torch.tensor(np.array(reward_list[  i:i+chunk_size            ]), dtype=torch.float)  )
                future_states_list.append(        torch.tensor(np.array(state_list [ i:i+chunk_size +1            ]), dtype=torch.float)  )
        else:
            for i in range(len(reward_list[:])):
                present_state_list.append(       torch.tensor(np.array(state_list [ i                        ]), dtype=torch.float)  )
                future_actions_list.append(      torch.tensor(np.array(action_list[ i   : i+chunk_size       ]), dtype=torch.float)  )
                future_rewards_list.append(       torch.tensor(np.array(reward_list[ i:i+chunk_size           ]), dtype=torch.float)  )
                future_states_list .append(       torch.tensor(np.array(state_list [ i:i+chunk_size +1            ]), dtype=torch.float)  )

    mask_value = 0
    future_actions_list =  [F.pad(torch.tensor(arr), pad=(0, 0, 0, pad_size - torch.tensor(arr).size(0)), mode='constant', value= mask_value) for arr in future_actions_list]
    future_rewards_list =  [F.pad(torch.tensor(arr), pad=(0, 0, 0, pad_size - torch.tensor(arr).size(0)), mode='constant', value= mask_value) for arr in future_rewards_list]
    future_states_list  =  [F.pad(torch.tensor(arr), pad=(0, 0, 0, pad_size - torch.tensor(arr).size(0)), mode='constant', value= mask_value) for arr in future_states_list]
    pad_size_list = [(pad_size - torch.tensor(arr).size(0)) for arr in future_states_list]

    present_state_tensors  = torch.stack( present_state_list  ).to(device)
    future_actions_tensors = torch.stack( future_actions_list ).to(device)
    future_rewards_tensors  = torch.stack( future_rewards_list  ).to(device)
    future_states_tensors   = torch.stack( future_states_list   ).to(device)
    pad_size_tensors  = torch.tensor(pad_size_list).to(device)
    # row_mask = torch.all(future_actions_tensors == mask_value, dim = -1)
    # mask_tensors = torch.zeros_like(future_actions_tensors, dtype = torch.bool)
    # mask_tensors[row_mask] = True

    return present_state_tensors, future_actions_tensors, future_rewards_tensors, future_states_tensors, pad_size_tensors#, mask_tensors




def obtain_TD_error(model,
                    data_loader,
                    device
                    ):

    for present_state_tensor, future_actions_tensor, future_rewards_tensor, future_states_tensor, pad_size_tensor in data_loader:

 
        range_tensor = torch.arange(future_actions_tensor.size(1)).to(device)  
        pad_tensor = (range_tensor < pad_size_tensor.unsqueeze(1)).int() .unsqueeze(-1).to(device)


        model.eval()

        loss_function                 = model.loss_function_
        output_reward, output_state   = model(present_state_tensor, future_actions_tensor)
        total_loss                    = loss_function(output_reward * pad_tensor, future_rewards_tensor * pad_tensor) 
        total_loss                    = torch.sum(torch.abs(total_loss), dim=(1, 2))
        TD_error                      = np.array(total_loss.detach().cpu())

    return TD_error




def update_model(iteration_for_learning,
                 long_term_present_state_tensors ,
                 long_term_future_actions_tensors,
                 long_term_future_rewards_tensors ,
                 long_term_future_states_tensors  ,
                 long_term_pad_size_tensors          ,
                 model,
                 PER_epsilon,
                 PER_exponent,
                 device):
    
    dataset     = TensorDataset(long_term_present_state_tensors, long_term_future_actions_tensors, long_term_future_rewards_tensors, long_term_future_states_tensors, long_term_pad_size_tensors)
    batch_size  = len(long_term_present_state_tensors)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    for _ in range(iteration_for_learning):


        TD_error         = obtain_TD_error(model, 
                                           data_loader,
                                           device )
        TD_error         =(TD_error + PER_epsilon) ** PER_exponent
        TD_error_p       = TD_error / np.sum(TD_error)
        index            = np.random.choice(range(batch_size), 
                                            p=TD_error_p, 
                                            size=1,
                                            replace=True)[0]

        present_state_tensor  = long_term_present_state_tensors [index].unsqueeze(0)
        future_actions_tensor = long_term_future_actions_tensors[index].unsqueeze(0)
        future_rewards_tensor  = long_term_future_rewards_tensors [index].unsqueeze(0)
        future_states_tensor   = long_term_future_states_tensors  [index].unsqueeze(0)
        pad_size_tensor           = long_term_pad_size_tensors          [index].unsqueeze(0)

        range_tensor = torch.arange(future_actions_tensor.size(1))  .to(device)
        pad_tensor = (range_tensor < pad_size_tensor.unsqueeze(1)).int() .unsqueeze(-1).to(device)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        output_reward, output_state = model(present_state_tensor, future_actions_tensor)
        total_loss                  = loss_function(output_reward * pad_tensor, future_rewards_tensor* pad_tensor) + loss_function(output_state * pad_tensor, future_states_tensor * pad_tensor)
        total_loss.backward()     # get grad

        selected_optimizer.step() # update params

    return model




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)


