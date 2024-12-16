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




import concurrent.futures
import hashlib


def initialize_pre_activated_action(init, noise_t, noise_r, shape):
    input = 0
    if   init == "random_uniform":
        for _ in range(noise_t):
            input += np.random.uniform(low=-noise_r, high=noise_r, size=shape) 
    elif init == "random_normal":
        for _ in range(noise_t):
            input += np.random.normal(loc=0.0, scale= noise_r, size= shape ) 
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
                                present_state,
                                pre_activated_future_action,
                                desired_reward,
                                beta,
                                device):

    present_state, pre_activated_future_action, desired_reward = present_state.to(device), pre_activated_future_action.to(device), desired_reward.to(device)

    model_list_copy   = copy.deepcopy(model_list)

    time_size         = pre_activated_future_action.size(1)

    for i in range(iteration_for_deducing):

        index         = np.random.randint(len(model_list_copy))
        model         = model_list_copy[index]
        tgt_indx      = np.random.randint(time_size) + 1

        future_action = torch.sigmoid(pre_activated_future_action[:, :tgt_indx])

        model.train()
        future_action = future_action.detach().requires_grad_(True)
        for param in model.parameters():
            param.requires_grad = False

        loss_function       = model.loss_function
        output_reward, _    = model(present_state, future_action)
        total_loss          = loss_function(output_reward, desired_reward[:, :tgt_indx])
        total_loss.backward() # get grad

        pre_activated_future_action[:, :tgt_indx] -= future_action.grad * (1 - future_action) * future_action * beta # update params
    
    return pre_activated_future_action




# def process_chunk(tuple_arg):
# 
#     state_list, action_list, reward_list, window_size, device = tuple_arg
# 
#     present_state_list = []
#     future_action_list = []
#     future_reward_list = []
#     future_state_list  = []
# 
#     if window_size != 1:
#         for j in range(len(reward_list[:-window_size+1])):
#             present_state_list.append(      torch.tensor(np.array(state_list [ j                               ]), dtype=torch.float).to(device)  )
#             future_action_list.append(      torch.tensor(np.array(action_list[ j     : j + window_size         ]), dtype=torch.float).to(device)  )
#             future_reward_list.append(      torch.tensor(np.array(reward_list[ j     : j + window_size         ]), dtype=torch.float).to(device)  )
#             future_state_list.append(       torch.tensor(np.array(state_list [ j + 1 : j + window_size + 1     ]), dtype=torch.float).to(device)  )
#     else:
#         for j in range(len(reward_list[:])):
#             present_state_list.append(      torch.tensor(np.array(state_list [ j                               ]), dtype=torch.float).to(device)  )
#             future_action_list.append(      torch.tensor(np.array(action_list[ j     : j + window_size         ]), dtype=torch.float).to(device)  )
#             future_reward_list.append(      torch.tensor(np.array(reward_list[ j     : j + window_size         ]), dtype=torch.float).to(device)  )
#             future_state_list.append(       torch.tensor(np.array(state_list [ j + 1 : j + window_size + 1     ]), dtype=torch.float).to(device)  )
# 
#     return present_state_list, future_action_list, future_reward_list, future_state_list
# 
# def sequentialize_(state_list, action_list, reward_list, max_window_size, device):
# 
#     if max_window_size > len(state_list[:-1]):
#         max_window_size = len(state_list[:-1])
#     else:
#       pass
# 
#     chunk = [(state_list, action_list, reward_list, i + 1, device) for i in range(max_window_size)]
# 
#     with mp.Pool(max_window_size) as pool:
# 
#         processed_chunks = pool.map(process_chunk, chunk)
# 
#     present_state_list = []
#     future_action_list = []
#     future_reward_list = []
#     future_state_list  = []
#     
#     # Iterate over processed chunks and extend the result lists
#     for processed_chunk in processed_chunks:
#         present_state_list.extend(processed_chunk[0])  
#         future_action_list.extend(processed_chunk[1])  
#         future_reward_list.extend(processed_chunk[2])  
#         future_state_list.extend (processed_chunk[3])  
#     
#     return present_state_list, future_action_list, future_reward_list, future_state_list




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
        chunk_size_ = j + 1
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




def hash_tensor(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

def fast_check_with_hash(hash_1d, hash_2d):
    return hash_1d not in hash_2d

def update_long_term_experience_buffer(present_state_tensor_dict, 
                                       future_action_tensor_dict,
                                       future_reward_tensor_dict, 
                                       future_state_tensor_dict,
                                       present_state_hash_list, 
                                       future_action_hash_list, 
                                       future_reward_hash_list, 
                                       future_state_hash_list,
                                       present_state_list,
                                       future_action_list,
                                       future_reward_list,
                                       future_state_list ):

    for i in range(len(present_state_list)):
        length             = len(future_action_list[i])
        present_state      = present_state_list  [i]
        future_action      = future_action_list  [i]
        future_reward      = future_reward_list  [i]
        future_state       = future_state_list   [i]
        present_state_hash = hash_tensor(present_state)
        future_action_hash = hash_tensor(future_action)
        future_reward_hash = hash_tensor(future_reward)
        future_state_hash  = hash_tensor(future_state )
        if  fast_check_with_hash(present_state_hash  , present_state_hash_list) or   \
            fast_check_with_hash(future_action_hash  , future_action_hash_list) or   \
            fast_check_with_hash(future_reward_hash  , future_reward_hash_list) or   \
            fast_check_with_hash(future_state_hash   , future_state_hash_list ) :
            present_state_tensor_dict  [length] = torch.cat((present_state_tensor_dict  [length],    present_state.unsqueeze(0) ), dim=0)
            future_action_tensor_dict  [length] = torch.cat((future_action_tensor_dict  [length],    future_action.unsqueeze(0) ), dim=0)
            future_reward_tensor_dict  [length] = torch.cat((future_reward_tensor_dict  [length],    future_reward.unsqueeze(0) ), dim=0)
            future_state_tensor_dict   [length] = torch.cat((future_state_tensor_dict   [length],    future_state .unsqueeze(0) ), dim=0)
            present_state_hash_list    .append( present_state_hash  )
            future_action_hash_list    .append( future_action_hash  )
            future_reward_hash_list    .append( future_reward_hash  )
            future_state_hash_list     .append( future_state_hash   )

    return present_state_tensor_dict, future_action_tensor_dict, future_reward_tensor_dict, future_state_tensor_dict,\
           present_state_hash_list, future_action_hash_list, future_reward_hash_list, future_state_hash_list




def obtain_TD_error(model,
                    present_state_tensor,
                    future_action_tensor,
                    future_reward_tensor,
                    future_state_tensor 
                    ):

    dataset      = TensorDataset(present_state_tensor,
                                 future_action_tensor,
                                 future_reward_tensor,
                                 future_state_tensor )
    data_loader  = DataLoader(dataset, batch_size = len(dataset), shuffle=False)

    for present_state, future_action, future_reward, future_state in data_loader:

        model.eval()

        loss_function                 = model.loss_function_
        output_reward, output_state   = model(present_state, future_action)
        total_loss                    = loss_function(output_reward, future_reward) 
        total_loss                    = torch.sum(torch.abs(total_loss), dim=(1, 2))

        TD_error                      = total_loss.detach()

    return TD_error




def update_model(iteration_for_learning,
                 present_state_tensor_dict,
                 future_action_tensor_dict,
                 future_reward_tensor_dict,
                 future_state_tensor_dict ,
                 model,
                 PER_epsilon ,
                 PER_exponent,
                 device):

    PER_epsilon  = torch.tensor(PER_epsilon  ).to(device)
    PER_exponent = torch.tensor(PER_exponent ).to(device)

    for _ in range(iteration_for_learning):

        random_key           = random.choice(list(present_state_tensor_dict.keys()))
        present_state_tensor = present_state_tensor_dict[random_key]
        future_action_tensor = future_action_tensor_dict[random_key]
        future_reward_tensor = future_reward_tensor_dict[random_key]
        future_state_tensor  = future_state_tensor_dict [random_key]

        TD_error             = obtain_TD_error (model, 
                                                present_state_tensor  ,
                                                future_action_tensor  ,
                                                future_reward_tensor  ,
                                                future_state_tensor   )
        TD_error             =(TD_error + PER_epsilon) ** PER_exponent
        TD_error_p           = TD_error / torch.sum(TD_error)
        index                = torch.multinomial(TD_error_p, 1, replacement = True)[0]

        present_state        = present_state_tensor [index].unsqueeze(0)
        future_action        = future_action_tensor [index].unsqueeze(0)
        future_reward        = future_reward_tensor [index].unsqueeze(0)
        future_state         = future_state_tensor  [index].unsqueeze(0)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        output_reward, output_state = model(present_state, future_action)
        total_loss                  = loss_function(output_reward, future_reward) + loss_function(output_state, future_state)
        total_loss.backward()     # get grad

        selected_optimizer.step() # update params

    return model




def update_model_parallel(iteration_for_learning,
                          present_state_tensor_dict,
                          future_action_tensor_dict,
                          future_reward_tensor_dict,
                          future_state_tensor_dict ,
                          model_list,  # List of models
                          PER_epsilon,
                          PER_exponent,
                          device):
    """
    Parallel training of multiple models on the same GPU.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_model = {
            executor.submit(update_model, 
                            iteration_for_learning,
                            present_state_tensor_dict,
                            future_action_tensor_dict,
                            future_reward_tensor_dict,
                            future_state_tensor_dict ,
                            model,
                            PER_epsilon,
                            PER_exponent,
                            device): model
            for model in model_list
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            results.append(future.result())
    
    return results




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)


