
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




def obtain_model_error(model_list,
                       state,
                       pre_activated_future_action,
                       desired_reward):

    model_error_list = []

    for model in model_list:

        future_action = torch.sigmoid(pre_activated_future_action)

        model.train()
        future_action = future_action.clone().detach().requires_grad_(True)
        if future_action.grad is not None:
            future_action.grad.zero_()
        for param in model.parameters():
            param.requires_grad = False

        loss_function      = model.loss_function
        output_reward, _   = model(state, future_action)
        total_loss         = loss_function(output_reward[:, -1, :], desired_reward)
        model_error_list.append(total_loss.detach().cpu())

    return np.array(model_error_list)




def update_pre_activated_action_(iteration_for_deducing,
                                model_list,
                                state,
                                pre_activated_future_action,
                                desired_reward,
                                beta,
                                device):

    state, pre_activated_future_action, desired_reward = state.to(device), pre_activated_future_action.to(device), desired_reward.to(device)
    
    model_list_copy = copy.deepcopy(model_list)

    for _ in range(iteration_for_deducing):




        model_error      = obtain_model_error(model_list_copy, 
                                              state, 
                                              pre_activated_future_action, 
                                              desired_reward)
        model_error      =(model_error + 0.000001) ** (-1)
        model_error_p    = model_error / np.sum(model_error)
        index            = np.random.choice(range(len(model_list_copy)), 
                                            p=model_error_p, 
                                            size=1,
                                            replace=True)[0]




        model            = model_list_copy[index]




        future_action = torch.sigmoid(pre_activated_future_action)

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




def update_pre_activated_action(iteration_for_deducing,
                                model_list,
                                state,
                                pre_activated_future_action,
                                desired_reward,
                                beta,
                                device):

    state, pre_activated_future_action, desired_reward = state.to(device), pre_activated_future_action.to(device), desired_reward.to(device)

    model_list_copy = copy.deepcopy(model_list)

    for _ in range(iteration_for_deducing):




        index            = np.random.randint(len(model_list_copy))
        model            = model_list_copy[index]




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




def group_list(replay_list_all):
    # Using defaultdict to group tensors by size (fast and efficient)
    replay_dict_all = defaultdict(list)
    # Group tuples by the shape of their tensors
    for tensors in replay_list_all:
        # Use the shape of the first tensor in the tuple as the key for grouping (you can change this to fit your needs)
        shape_key = tuple(tensor.shape for tensor in tensors)
        replay_dict_all[shape_key].append(tensors)
    return replay_dict_all




def obtain_TD_error(model,
                    long_term_sequentialized_state   ,
                    long_term_sequentialized_action  ,
                    long_term_sequentialized_reward  ,
                    long_term_sequentialized_n_state ,
                    device):




    long_term_sequentialized_state_tensor   = torch.tensor(np.array(long_term_sequentialized_state  ), dtype=torch.float).to(device)  
    long_term_sequentialized_action_tensor  = torch.tensor(np.array(long_term_sequentialized_action ), dtype=torch.float).to(device)  
    long_term_sequentialized_reward_tensor  = torch.tensor(np.array(long_term_sequentialized_reward ), dtype=torch.float).to(device)  
    long_term_sequentialized_n_state_tensor = torch.tensor(np.array(long_term_sequentialized_n_state), dtype=torch.float).to(device)  

    dataset      = TensorDataset(long_term_sequentialized_state_tensor     ,
                                 long_term_sequentialized_action_tensor    ,
                                 long_term_sequentialized_reward_tensor    ,
                                 long_term_sequentialized_n_state_tensor   )
    data_loader  = DataLoader(dataset, batch_size = len(dataset), shuffle=False)

    for state, future_action, future_reward, future_state in data_loader:

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function                 = model.loss_function_
        output_reward, output_state   = model(state, future_action)
        total_loss                    = loss_function(output_reward, future_reward)
        total_loss                    = torch.sum(torch.abs(total_loss), dim=(1, 2))
        TD_error                      = np.array(total_loss.detach().cpu())

    return TD_error




def update_model(iteration_for_learning,
                 list_tuple,
                 model,
                 PER_epsilon,
                 PER_exponent,
                 device):


    dict_tuple = group_list(list_tuple)
    

    for _ in range(int(iteration_for_learning)):


        list_tuple       = random.choice(list(dict_tuple.values())) # list_tuple is a list of tuples of state, action and reward of the same time_size
        list_tuple       = list(zip(*list_tuple))




        TD_error         = obtain_TD_error(model, 
                                           list_tuple[0] ,
                                           list_tuple[1] ,
                                           list_tuple[2] ,
                                           list_tuple[3] ,
                                           device)
        TD_error         =(TD_error + PER_epsilon) ** PER_exponent
        TD_error_p       = TD_error / np.sum(TD_error)
        index            = np.random.choice(range(len(list_tuple[0])), 
                                            p=TD_error_p, 
                                            size=1,
                                            replace=True)[0]




        state            = list_tuple[0][index].unsqueeze(0).to(device)
        future_action    = list_tuple[1][index].unsqueeze(0).to(device)
        future_reward    = list_tuple[2][index].unsqueeze(0).to(device)
        future_state     = list_tuple[3][index].unsqueeze(0).to(device)




        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        output_reward, output_state = model(state, future_action)
        total_loss                  = loss_function(output_reward, future_reward) + loss_function(output_state, future_state)
        total_loss.backward()     # get grad

        selected_optimizer.step() # update params

    return model




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)




