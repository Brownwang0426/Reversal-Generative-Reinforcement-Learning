
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




def update_pre_activated_action(epoch_for_deducing,
                                model_list,
                                state,
                                pre_activated_future_action,
                                desired_reward,
                                beta,
                                device):

    state, pre_activated_future_action, desired_reward = state.to(device), pre_activated_future_action.to(device), desired_reward.to(device)

    model_list_copy = copy.deepcopy(model_list)

    for _ in range(epoch_for_deducing):

        random.shuffle(model_list_copy)

        for model in model_list_copy:

            future_action    = torch.sigmoid(pre_activated_future_action)

            model.train()
            future_action = future_action.clone().detach().requires_grad_(True)
            if future_action.grad is not None:
                future_action.grad.zero_()
            for param in model.parameters():
                param.requires_grad = False

            loss_function       = model.loss_function
            output_reward, _    = model(state, future_action)
            total_loss          = loss_function(output_reward[:, -1, :], desired_reward)
            total_loss.backward() # get grad

            pre_activated_future_action -= future_action.grad * (1 - future_action) * future_action * beta # update params

    return pre_activated_future_action




def sequentialize(state_list, action_list, reward_list, chunk_size):

    present_state_list = []
    future_action_list = []
    future_reward_list = []
    future_state_list  = []

    if chunk_size > len(state_list[:-1]):
        chunk_size = len(state_list[:-1])
    else:
      pass
    
    if chunk_size != 1:
        for i in range(len(reward_list[:-chunk_size+1])):
            present_state_list.append(      torch.tensor(np.array(state_list [ i                        ]), dtype=torch.float)  )
            future_action_list.append(      torch.tensor(np.array(action_list[ i   : i+chunk_size       ]), dtype=torch.float)  )
            future_reward_list.append(      torch.tensor(np.array(reward_list[ i   : i+chunk_size       ]), dtype=torch.float)  )
            future_state_list.append(       torch.tensor(np.array(state_list [ i+1 : i+chunk_size+1     ]), dtype=torch.float)  )
    else:
        for i in range(len(reward_list[:])):
            present_state_list.append(      torch.tensor(np.array(state_list [ i                        ]), dtype=torch.float)  )
            future_action_list.append(      torch.tensor(np.array(action_list[ i   : i+chunk_size       ]), dtype=torch.float)  )
            future_reward_list.append(      torch.tensor(np.array(reward_list[ i   : i+chunk_size       ]), dtype=torch.float)  )
            future_state_list.append(       torch.tensor(np.array(state_list [ i+1 : i+chunk_size+1     ]), dtype=torch.float)  )

    return present_state_list, future_action_list, future_reward_list, future_state_list




def obtain_TD_error(model_list,
                    list_tuple,
                    PER_epsilon,
                    PER_exponent,
                    device):

    list_tuple     = list(zip(*list_tuple))

    state_tuple    = list_tuple[0]
    action_tuple   = list_tuple[1]
    reward_tuple   = list_tuple[2]
    n_state_tuple  = list_tuple[3]

    state_tensor   = torch.tensor(np.array(state_tuple  ), dtype=torch.float).to(device)  
    action_tensor  = torch.tensor(np.array(action_tuple ), dtype=torch.float).to(device)  
    reward_tensor  = torch.tensor(np.array(reward_tuple ), dtype=torch.float).to(device)  
    n_state_tensor = torch.tensor(np.array(n_state_tuple), dtype=torch.float).to(device)  

    dataset        = TensorDataset(state_tensor     ,
                                   action_tensor    ,
                                   reward_tensor    ,
                                   n_state_tensor   )
    data_loader    = DataLoader(dataset, batch_size = len(dataset), shuffle=False)

    TD_error_all   = 0
    for model in model_list:
        
        for state, future_action, future_reward, future_state in data_loader:

            model.train()
            selected_optimizer = model.selected_optimizer
            selected_optimizer.zero_grad()

            loss_function                 = model.loss_function_
            output_reward, output_state   = model(state, future_action)
            total_loss                    = loss_function(output_reward, future_reward)
            total_loss                    = torch.sum(torch.abs(total_loss), dim=(1, 2))
            TD_error                      = np.array(total_loss.detach().cpu()) 

        TD_error_all += TD_error
    TD_error_all      =(TD_error_all + PER_epsilon) ** PER_exponent
    TD_error_p        = TD_error_all / np.sum(TD_error_all)

    index = np.random.choice(range(len(dataset)), 
                             p=TD_error_p, 
                             size=1,
                             replace=True)[0]

    return index 




def update_model(epoch_for_learning,
                 model_list,
                 list_tuple,
                 t_index,
                 device):

    list_tuple           = list(zip(*list_tuple))

    state_tuple          = list_tuple[0]
    action_tuple         = list_tuple[1]
    reward_tuple         = list_tuple[2]
    n_state_tuple        = list_tuple[3]

    for _ in range(int(epoch_for_learning)):

        random.shuffle(model_list)

        for model in model_list:

            state            = state_tuple   [t_index].unsqueeze(0).to(device)
            future_action    = action_tuple  [t_index].unsqueeze(0).to(device)
            future_reward    = reward_tuple  [t_index].unsqueeze(0).to(device)
            future_state     = n_state_tuple [t_index].unsqueeze(0).to(device)

            model.train()
            selected_optimizer = model.selected_optimizer
            selected_optimizer.zero_grad()

            loss_function               = model.loss_function
            output_reward, output_state = model(state, future_action)
            total_loss                  = loss_function(output_reward, future_reward) + loss_function(output_state, future_state)
            total_loss.backward()     # get grad

            selected_optimizer.step() # update params

    return model_list




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)


