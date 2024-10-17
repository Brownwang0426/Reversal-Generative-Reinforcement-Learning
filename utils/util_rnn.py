
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
#         model_error_list.append(total_loss)
# 
#     return torch.tensor(model_error_list)




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




        # model_error      = obtain_model_error(model_list_copy, state, pre_activated_future_action, desired_reward)
        # index            = np.argmin(model_error.cpu().numpy())
        index            = np.random.randint(len(model_list_copy))




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




def sequentialize(short_term_state_list, short_term_action_list, short_term_reward_list, chunk_size):

    short_term_present_state_list = []
    short_term_future_action_list = []
    short_term_future_reward_list = []
    short_term_future_state_list  = []

    if chunk_size > len(short_term_state_list[:-1]):
        chunk_size = len(short_term_state_list[:-1])
    else:
      pass

    if chunk_size != 1:
        for i in range(len(short_term_reward_list[:-chunk_size+1])):
            short_term_present_state_list.append(      short_term_state_list [ i                        ]  )
            short_term_future_action_list.append(      short_term_action_list[ i   : i+chunk_size       ]  )
            short_term_future_reward_list.append(      short_term_reward_list[ i   : i+chunk_size       ]  )
            short_term_future_state_list.append(       short_term_state_list [ i+1 : i+chunk_size+1     ]  )
    else:
        for i in range(len(short_term_reward_list[:])):
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
                            num_heads,
                            device):

    # Convert lists to tensors directly on the desired device and data type
    short_term_state_tensor         = torch.tensor(np.array(short_term_state_list), dtype=torch.float).to(device)
    short_term_future_action_tensor = torch.tensor(np.array(short_term_future_action_list), dtype=torch.float).to(device)
    short_term_future_reward_tensor = torch.tensor(np.array(short_term_future_reward_list), dtype=torch.float).to(device)
    short_term_future_state_tensor  = torch.tensor(np.array(short_term_future_state_list), dtype=torch.float).to(device)

    return short_term_state_tensor, short_term_future_action_tensor, short_term_future_reward_tensor, short_term_future_state_tensor




def obtain_TD_error(model,
                    data_loader):

    for state, future_action, future_reward, future_state in data_loader:

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function                 = model.loss_function_
        output_reward, output_state   = model(state, future_action)
        total_loss_1                  = loss_function(output_reward, future_reward).detach()
        total_loss_1                  = torch.sum(torch.abs(total_loss_1), dim=(1, 2))
        # total_loss_2                  = loss_function(output_state, future_state).detach()
        # total_loss_2                  = torch.sum(torch.abs(total_loss_2), dim=(0, 2, 3))

    # Since TD error amis to select samples that are surprising to the agent and 
    # we think the term "surprising" might have more to do with reward other than states, 
    # therefore we leave only total_loss_1 (error for reward) for TD error.
    # However, you may try adding back total_loss_2 to see what will happen. But in our experience, it is not a good idea...
    TD_error = total_loss_1 # + total_loss_2
    return TD_error 




def update_model(iteration_for_learning,
                 dataset,
                 data_loader,
                 model,
                 batch_size,
                 PER_epsilon,
                 PER_exponent):


    for _ in range(iteration_for_learning):




        TD_error         = obtain_TD_error(model, data_loader)



        
        # TD_error         =(TD_error.cpu().numpy() + PER_epsilon) ** PER_exponent
        # TD_error_p       = TD_error / np.sum(TD_error)
        # index            = np.random.choice(range(len(dataset)), 
        #                                     p=TD_error_p, 
        #                                     size=1)
        index            = np.argmax(TD_error.cpu().numpy())




        pair             = dataset[index]
        state            = pair[0].unsqueeze(0)
        future_action    = pair[1].unsqueeze(0)
        future_reward    = pair[2].unsqueeze(0)
        future_state     = pair[3].unsqueeze(0)




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


