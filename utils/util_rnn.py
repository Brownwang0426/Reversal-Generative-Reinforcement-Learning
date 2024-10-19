
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




# def obtain_model_error(model_list,
#                        state,
#                        pre_activated_future_action,
#                        desired_reward):
# 
#     model_error_list = []
# 
#     for model in model_list:
# 
#         future_action = torch.sigmoid(pre_activated_future_action)
# 
#         model.train()
#         future_action = future_action.clone().detach().requires_grad_(True)
#         if future_action.grad is not None:
#             future_action.grad.zero_()
#         for param in model.parameters():
#             param.requires_grad = False
# 
#         loss_function      = model.loss_function
#         output_reward, _   = model(state, future_action)
#         total_loss         = loss_function(output_reward[:, -1, :], desired_reward)
#         model_error_list.append(total_loss.detach().cpu())
# 
#     return np.array(model_error_list)




# def update_pre_activated_action(iteration_for_deducing,
#                                 model_list,
#                                 state,
#                                 pre_activated_future_action,
#                                 desired_reward,
#                                 beta,
#                                 device):
# 
#     state, pre_activated_future_action, desired_reward = state.to(device), pre_activated_future_action.to(device), desired_reward.to(device)
#     
#     model_list_copy = copy.deepcopy(model_list)
# 
# 
# 
# 
#     model_error      = obtain_model_error(model_list_copy, 
#                                           state, 
#                                           pre_activated_future_action, 
#                                           desired_reward)
#     model_error      =(model_error + 0.000001) ** (-1)
#     model_error_p    = model_error / np.sum(model_error)
#     index_list       = np.random.choice(range(len(model_list_copy)), 
#                                         p=model_error_p, 
#                                         size=iteration_for_deducing,
#                                         replace=True)
# 
# 
#     
# 
#     for i, index in enumerate(index_list):
# 
# 
# 
# 
#         model            = model_list_copy[index]
# 
# 
# 
# 
#         future_action = torch.sigmoid(pre_activated_future_action)
# 
#         model.train()
#         future_action = future_action.clone().detach().requires_grad_(True)
#         if future_action.grad is not None:
#             future_action.grad.zero_()
#         for param in model.parameters():
#             param.requires_grad = False
# 
#         loss_function       = model.loss_function
#         output_reward, _    = model(state, future_action)
#         total_loss          = loss_function(output_reward[:, -1, :], desired_reward)
#         total_loss.backward() # get grad
# 
#         pre_activated_future_action -= future_action.grad * (1 - future_action) * future_action * beta # update params
# 
#     return pre_activated_future_action




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




def sequentialize(state_list, action_list, reward_list, chunk_size, device):

    present_state_list = []
    future_action_list = []
    future_reward_list = []
    future_state_list  = []

    if chunk_size > len(state_list[:-1]):
        chunk_size = len(state_list[:-1])
    else:
      pass
    
    for j in range(chunk_size):
        chunk_size_ = 1 + j
        if chunk_size_ != 1:
            for i in range(len(reward_list[:-chunk_size_+1])):
                present_state_list.append(      torch.tensor(np.array(state_list [ i                         ]), dtype=torch.float).to(device)  )
                future_action_list.append(      torch.tensor(np.array(action_list[ i   : i+chunk_size_       ]), dtype=torch.float).to(device)  )
                future_reward_list.append(      torch.tensor(np.array(reward_list[ i   : i+chunk_size_       ]), dtype=torch.float).to(device)  )
                future_state_list.append(       torch.tensor(np.array(state_list [ i+1 : i+chunk_size_+1     ]), dtype=torch.float).to(device)  )
        else:
            for i in range(len(reward_list[:])):
                present_state_list.append(      torch.tensor(np.array(state_list [ i                         ]), dtype=torch.float).to(device)  )
                future_action_list.append(      torch.tensor(np.array(action_list[ i   : i+chunk_size_       ]), dtype=torch.float).to(device)  )
                future_reward_list.append(      torch.tensor(np.array(reward_list[ i   : i+chunk_size_       ]), dtype=torch.float).to(device)  )
                future_state_list.append(       torch.tensor(np.array(state_list [ i+1 : i+chunk_size_+1     ]), dtype=torch.float).to(device)  )

    return present_state_list, future_action_list, future_reward_list, future_state_list




def obtain_TD_error(model,
                    long_term_sequentialized_state_list   ,
                    long_term_sequentialized_action_list  ,
                    long_term_sequentialized_reward_list  ,
                    long_term_sequentialized_n_state_list ):

    TD_error = list()

    for i in range(len(long_term_sequentialized_action_list)):




        state            = long_term_sequentialized_state_list   [i].unsqueeze(0)
        future_action    = long_term_sequentialized_action_list  [i].unsqueeze(0)
        future_reward    = long_term_sequentialized_reward_list  [i].unsqueeze(0)
        future_state     = long_term_sequentialized_n_state_list [i].unsqueeze(0)




        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        TD_error_type = 5
        if TD_error_type == 1:
            loss_function                 = model.loss_function
            output_reward, output_state   = model(state, future_action)
            total_loss_1                  = loss_function(output_reward, future_reward).detach()
            total_loss_2                  = loss_function(output_state, future_state).detach()
            TD_error.append(total_loss_1.cpu() + total_loss_2.cpu())   
        elif TD_error_type == 2:
            loss_function                 = model.loss_function
            output_reward, output_state   = model(state, future_action)
            total_loss_1                  = loss_function(output_reward, future_reward).detach()
            total_loss_1                  = total_loss_1 / (future_reward.size(1) * future_reward.size(2))
            total_loss_2                  = loss_function(output_state, future_state).detach()
            total_loss_2                  = total_loss_2 / (future_state.size(1) * future_state.size(2))
            TD_error.append(total_loss_1.cpu() + total_loss_2.cpu())   
        elif TD_error_type == 3:
            loss_function                 = model.loss_function
            output_reward, output_state   = model(state, future_action)
            total_loss_1                  = loss_function(output_reward[:, -1, :], future_reward[:, -1, :]).detach()
            TD_error.append(total_loss_1.cpu())                      
        elif TD_error_type == 4:
            loss_function                 = model.loss_function
            output_reward, output_state   = model(state, future_action)
            total_loss_1                  = loss_function(output_reward, future_reward).detach()
            TD_error.append(total_loss_1.cpu())   
        elif TD_error_type == 5:
            loss_function                 = model.loss_function
            output_reward, output_state   = model(state, future_action)
            total_loss_1                  = loss_function(output_reward, future_reward).detach()
            total_loss_1                  = total_loss_1 / future_reward.size(1)
            TD_error.append(total_loss_1.cpu())   

    return np.array(TD_error)




def update_model(iteration_for_learning,
                 long_term_sequentialized_state_list   ,
                 long_term_sequentialized_action_list  ,
                 long_term_sequentialized_reward_list  ,
                 long_term_sequentialized_n_state_list ,
                 model,
                 PER_epsilon,
                 PER_exponent):




    TD_error         = obtain_TD_error(model, 
                                       long_term_sequentialized_state_list   ,
                                       long_term_sequentialized_action_list  ,
                                       long_term_sequentialized_reward_list  ,
                                       long_term_sequentialized_n_state_list  )
    TD_error         =(TD_error + PER_epsilon) ** PER_exponent
    TD_error_p       = TD_error / np.sum(TD_error)
    index_list       = np.random.choice(range(len(long_term_sequentialized_action_list)), 
                                        p=TD_error_p, 
                                        size=iteration_for_learning,
                                        replace=True)




    for i, index in enumerate(index_list):




        state            = long_term_sequentialized_state_list   [index].unsqueeze(0)
        future_action    = long_term_sequentialized_action_list  [index].unsqueeze(0)
        future_reward    = long_term_sequentialized_reward_list  [index].unsqueeze(0)
        future_state     = long_term_sequentialized_n_state_list [index].unsqueeze(0)




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


