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




def update_pre_activated_action(iteration_for_deducing,
                                model_list,
                                state,
                                pre_activated_future_action,
                                desired_reward,
                                beta,
                                device):

    state, pre_activated_future_action, desired_reward = state.to(device), pre_activated_future_action.to(device), desired_reward.to(device)

    model_list_copy = copy.deepcopy(model_list)

    for i in range(iteration_for_deducing):

        index            = np.random.randint(len(model_list_copy))
        model            = model_list_copy[index]

        future_action    = torch.sigmoid(pre_activated_future_action)

        model.train()
        future_action = future_action.clone().detach().requires_grad_(True)
        if future_action.grad is not None:
            future_action.grad.zero_()
        for param in model.parameters():
            param.requires_grad = False

        tgt_indx            = np.random.randint(future_action.size(1))

        loss_function       = model.loss_function
        output_reward, _    = model(state, future_action)
        total_loss          = loss_function(output_reward[:, tgt_indx], desired_reward[:, tgt_indx])
        total_loss.backward() # get grad

        pre_activated_future_action[:, :tgt_indx+1] -= future_action.grad[:, :tgt_indx+1] * (1 - future_action[:, :tgt_indx+1]) * future_action[:, :tgt_indx+1] * beta # update params

    del model_list_copy
    gc.collect()
    torch.cuda.empty_cache()
    
    return pre_activated_future_action




def sequentialize(state_list, action_list, reward_list, chunk_size_):

    present_state_list = []
    future_action_list = []
    future_reward_list = []
    future_state_list  = []

    if chunk_size_ > len(state_list[:-1]):
        chunk_size_ = len(state_list[:-1])
    else:
      pass
    
    for j in range(chunk_size_):
        chunk_size = j + 1
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




def obtain_TD_error(model,
                    state_tensors   ,
                    action_tensors  ,
                    reward_tensors  ,
                    n_state_tensors 
                    ):

    dataset      = TensorDataset(state_tensors  ,
                                 action_tensors ,
                                 reward_tensors ,
                                 n_state_tensors)
    data_loader  = DataLoader(dataset, batch_size = len(dataset), shuffle=False)

    for state, future_action, future_reward, future_state in data_loader:

        model.eval()

        loss_function                 = model.loss_function_
        output_reward, output_state   = model(state, future_action)
        total_loss                    = loss_function(output_reward[:, -1], future_reward[:, -1]) 
        total_loss                    = torch.sum(torch.abs(total_loss), dim=(1))
        TD_error                      = np.array(total_loss.detach().cpu())

    return TD_error




def update_model(iteration_for_learning,
                 dict_list_state_tensors  ,
                 dict_list_action_tensors ,
                 dict_list_reward_tensors ,
                 dict_list_n_state_tensors,
                 model,
                 PER_epsilon,
                 PER_exponent,
                 device):

    for _ in range(iteration_for_learning):
        
        TD_error_list = list()
        key_list      = list()
        index_list    = list()

        for key in list(dict_list_state_tensors.keys()):

            state_tensors   = torch.stack(dict_list_state_tensors  [key]  ).to(device)  # 2d tensor [n,  ..., n]
            action_tensors  = torch.stack(dict_list_action_tensors [key]  ).to(device)  # 2d tensor [a,  ..., a]
            reward_tensors  = torch.stack(dict_list_reward_tensors [key]  ).to(device)  # 2d tensor [r,  ..., r]
            n_state_tensors = torch.stack(dict_list_n_state_tensors[key]  ).to(device)  # 2d tensor [ns, ..., ns]

            TD_error       = obtain_TD_error(model, 
                                             state_tensors    ,
                                             action_tensors   ,
                                             reward_tensors   ,
                                             n_state_tensors  )

            TD_error_list.extend(TD_error.tolist())
            key_list     .extend([key] * len(TD_error))
            index_list   .extend(list(range(len(TD_error))))

        TD_error         = np.array(TD_error_list)
        TD_error         =(TD_error + PER_epsilon) ** PER_exponent
        TD_error_p       = TD_error / np.sum(TD_error)
        index            = np.random.choice(range(len(key_list)), 
                                            p=TD_error_p, 
                                            size=1,
                                            replace=True)[0]
    
        state            = dict_list_state_tensors   [key_list[index]] [index_list[index]].unsqueeze(0).to(device)
        future_action    = dict_list_action_tensors  [key_list[index]] [index_list[index]].unsqueeze(0).to(device)
        future_reward    = dict_list_reward_tensors  [key_list[index]] [index_list[index]].unsqueeze(0).to(device)
        future_state     = dict_list_n_state_tensors [key_list[index]] [index_list[index]].unsqueeze(0).to(device)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        output_reward, output_state = model(state, future_action)
        total_loss                  = loss_function(output_reward[:, -1], future_reward[:, -1]) + loss_function(output_state, future_state)
        total_loss.backward()     # get grad

        selected_optimizer.step() # update params

    gc.collect()
    torch.cuda.empty_cache()

    return model




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)


