import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import minigrid

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

import dill

import warnings
warnings.filterwarnings('ignore')

import concurrent.futures
import hashlib

def load_performance_from_csv(filename='performance_log.csv'):
    performance_log = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            episode = int(row[0])  
            summed_reward = float(row[1])  
            performance_log.append((episode, summed_reward))
    return performance_log




def load_dicts_from_pickle(filename):
    with open(filename, 'rb') as file:
        dicts = dill.load(file)
    return dicts




def initialize_pre_activated_action(init, noise_t, noise_r, shape, device):
    input = 0
    if   init == "random_uniform":
        for _ in range(noise_t):
            input += (torch.rand(shape) * (noise_r * 2) - noise_r).to(device) 
    elif init == "random_normal":
        for _ in range(noise_t):
            input +=  torch.normal(mean=0, std=noise_r, size=shape).to(device) 
    elif (init == "glorot_uniform") or (init == "xavier_uniform"):
        for _ in range(noise_t):
            limit = np.sqrt(6 / (shape[1] + shape[1]))
            input += (torch.rand(shape) * (limit * 2) - limit).to(device) 
    elif (init == "glorot_normal" ) or (init == "xavier_normal"):
        for _ in range(noise_t):
            input += torch.normal(mean=0, std = np.sqrt(2 / (shape[1] + shape[1])) * noise_r, size=shape).to(device) 
    return input




def update_pre_activated_action(iteration_for_deducing,
                                model_list,
                                history_state,
                                history_action,
                                history_size,
                                present_state,
                                pre_activated_future_action,
                                desired_reward,
                                beta,
                                loss_scale,
                                device):

    history_state     = torch.stack(history_state [-history_size-1:-1], dim=0).unsqueeze(0).to(device)
    history_action    = torch.stack(history_action[-history_size:]    , dim=0).unsqueeze(0).to(device)

    present_state, pre_activated_future_action, desired_reward = present_state.to(device), pre_activated_future_action.to(device), desired_reward.to(device)

    time_size         = pre_activated_future_action.size(1)

    mask_1            = torch.full((1, 1, history_size * 2 + 1 + time_size, history_size * 2 + 1 + time_size), float("-inf"))
    mask_1            = torch.triu(mask_1, diagonal=1).to(device)

    for i in range(iteration_for_deducing):

        model         = random.choice(model_list)
        tgt_indx      = np.random.randint(time_size) 

        future_action = torch.sigmoid(pre_activated_future_action[:, :tgt_indx+1])
        value         = 0
        pad_size      = time_size - (tgt_indx + 1)
        pad           = (0, 0, 0, pad_size)
        future_action = F.pad(future_action, pad=pad, mode='constant', value= value)
        future_action = future_action.detach().requires_grad_(True)

        mask_2        = torch.tensor(history_size * 2 + 1 + tgt_indx).unsqueeze(0).to(device)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()
        
        loss_function       = model.loss_function
        output_reward       = model(history_state, history_action, present_state, future_action, (mask_1, mask_2))
        total_loss          = loss_function(output_reward, desired_reward) * (loss_scale ** tgt_indx)
        total_loss.backward() # get grad

        pre_activated_future_action[:, :tgt_indx+1] -= future_action.grad[:, :tgt_indx+1] * (1 - future_action[:, :tgt_indx+1]) * future_action[:, :tgt_indx+1] * beta # update params

    return pre_activated_future_action




def sequentialize(state_list, action_list, reward_list, history_size, time_size, device):

    max_time_size       = copy.deepcopy(time_size)

    history_state_list  = []
    history_action_list = []
    present_state_list  = []
    future_action_list  = []
    future_reward_list  = []
    mask_1_list         = []
    mask_2_list         = []

    if (history_size + time_size) > len(state_list[:-1]):
        time_size    = len(state_list[:-1]) - history_size
    else:
        pass

    for i in range(time_size):
        time_size_  = i + 1
        process_len = len(reward_list[:-(history_size + time_size_)+1])
        for j in range(process_len):
            history_state_list.append (      torch.stack(state_list [ j                : j+history_size                              ], dim=0)                                  .to(device) )
            history_action_list.append(      torch.stack(action_list[ j                : j+history_size                              ], dim=0)                                  .to(device) )
            present_state_list.append (                  state_list [ j+history_size                                                 ]                                          .to(device) )
            future_action_list.append (      torch.stack(action_list[ j+history_size   : j+history_size+time_size_                   ], dim=0)                                  .to(device) )
            future_reward_list.append (                  reward_list[                    j+history_size+time_size_ - 1               ]                                          .to(device) )
            mask_1_list       .append (       torch.triu(torch.full((1, history_size * 2 + 1 + max_time_size, history_size * 2 + 1 + max_time_size), float("-inf")), diagonal=1).to(device) )
            mask_2_list       .append (     torch.tensor(history_size * 2 + 1 + time_size_ - 1)                                                                                 .to(device) )

    for i, future_action in enumerate(future_action_list):

        value                 = 0
        pad_size              = max_time_size - future_action.size(0)
        pad                   = (0, 0, 0, pad_size)
        future_action_list[i] = F.pad(future_action, pad=pad, mode='constant', value= value)

    return history_state_list, history_action_list, present_state_list, future_action_list, future_reward_list, mask_1_list, mask_2_list




def hash_tensor(tensor):
    tensor = tensor.cpu()  # Make sure the tensor is on the CPU
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

def fast_check_with_hash(hash_1d, hash_2d):
    return hash_1d not in hash_2d

def update_long_term_experience_buffer(
                                       history_state_tensor_dict, 
                                       history_action_tensor_dict,
                                       present_state_tensor_dict, 
                                       future_action_tensor_dict,
                                       future_reward_tensor_dict, 
                                       mask_1_tensor_dict,
                                       mask_2_tensor_dict,
                                       history_state_hash_dict, 
                                       history_action_hash_dict, 
                                       present_state_hash_dict, 
                                       future_action_hash_dict, 
                                       future_reward_hash_dict, 
                                       mask_1_hash_dict,
                                       mask_2_hash_dict,
                                       history_state_list,
                                       history_action_list,
                                       present_state_list,
                                       future_action_list,
                                       future_reward_list,
                                       mask_1_list,
                                       mask_2_list):
    
    for i in range(len(present_state_list)):
        length              = len(future_action_list[i])
        history_state       = history_state_list [i]
        history_action      = history_action_list[i]
        present_state       = present_state_list [i]
        future_action       = future_action_list [i]
        future_reward       = future_reward_list [i]
        mask_1              = mask_1_list        [i]
        mask_2              = mask_2_list        [i]
        history_state_hash  = hash_tensor(history_state)
        history_action_hash = hash_tensor(history_action)
        present_state_hash  = hash_tensor(present_state)
        future_action_hash  = hash_tensor(future_action)
        future_reward_hash  = hash_tensor(future_reward)
        mask_1_hash         = hash_tensor(mask_1       )
        mask_2_hash         = hash_tensor(mask_2       )

        if  fast_check_with_hash(history_state_hash   , history_state_hash_dict [length]) or   \
            fast_check_with_hash(history_action_hash  , history_action_hash_dict[length]) or   \
            fast_check_with_hash(present_state_hash   , present_state_hash_dict [length]) or   \
            fast_check_with_hash(future_action_hash   , future_action_hash_dict [length]) or   \
            fast_check_with_hash(future_reward_hash   , future_reward_hash_dict [length]) or   \
            fast_check_with_hash(mask_1_hash          , mask_1_hash_dict        [length]) or   \
            fast_check_with_hash(mask_2_hash          , mask_2_hash_dict        [length]):

            history_state_tensor_dict  [length] = torch.cat((history_state_tensor_dict  [length],    history_state.unsqueeze(0) ), dim=0)
            history_action_tensor_dict [length] = torch.cat((history_action_tensor_dict [length],    history_action.unsqueeze(0)), dim=0)
            present_state_tensor_dict  [length] = torch.cat((present_state_tensor_dict  [length],    present_state.unsqueeze(0) ), dim=0)
            future_action_tensor_dict  [length] = torch.cat((future_action_tensor_dict  [length],    future_action.unsqueeze(0) ), dim=0)
            future_reward_tensor_dict  [length] = torch.cat((future_reward_tensor_dict  [length],    future_reward.unsqueeze(0) ), dim=0)
            mask_1_tensor_dict         [length] = torch.cat((mask_1_tensor_dict         [length],    mask_1       .unsqueeze(0) ), dim=0)
            mask_2_tensor_dict         [length] = torch.cat((mask_2_tensor_dict         [length],    mask_2       .unsqueeze(0) ), dim=0)
            history_state_hash_dict    [length] .append( history_state_hash  )
            history_action_hash_dict   [length] .append( history_action_hash )
            present_state_hash_dict    [length] .append( present_state_hash  )
            future_action_hash_dict    [length] .append( future_action_hash  )
            future_reward_hash_dict    [length] .append( future_reward_hash  )
            mask_1_hash_dict           [length] .append( mask_1_hash         )
            mask_2_hash_dict           [length] .append( mask_2_hash         )

    return history_state_tensor_dict, history_action_tensor_dict, present_state_tensor_dict, future_action_tensor_dict, future_reward_tensor_dict, mask_1_tensor_dict, mask_2_tensor_dict,\
           history_state_hash_dict, history_action_hash_dict, present_state_hash_dict, future_action_hash_dict, future_reward_hash_dict, mask_1_hash_dict, mask_2_hash_dict




def obtain_TD_error(model,
                    history_state_tensor,
                    history_action_tensor,
                    present_state_tensor,
                    future_action_tensor,
                    future_reward_tensor,
                    mask_1_tensor ,
                    mask_2_tensor
                    ):

    dataset      = TensorDataset(
                                 history_state_tensor,
                                 history_action_tensor,
                                 present_state_tensor,
                                 future_action_tensor,
                                 future_reward_tensor,
                                 mask_1_tensor,
                                 mask_2_tensor )
    data_loader  = DataLoader(dataset, batch_size = len(dataset), shuffle=False)

    for history_state, history_action, present_state, future_action, future_reward, mask_1, mask_2 in data_loader:

        model.eval()

        loss_function                 = model.loss_function_
        output_reward                 = model(history_state, history_action, present_state, future_action, (mask_1, mask_2))
        total_loss                    = loss_function(output_reward, future_reward) 

        total_loss                    = torch.sum(torch.abs(total_loss), dim=(1))
        TD_error                      = total_loss.detach()

    return TD_error




def clear_long_term_experience_buffer(
                                      history_state_tensor_dict, 
                                      history_action_tensor_dict,
                                      present_state_tensor_dict, 
                                      future_action_tensor_dict,
                                      future_reward_tensor_dict, 
                                      mask_1_tensor_dict,
                                      mask_2_tensor_dict,
                                      history_state_hash_dict, 
                                      history_action_hash_dict, 
                                      present_state_hash_dict, 
                                      future_action_hash_dict, 
                                      future_reward_hash_dict, 
                                      mask_1_hash_dict ,
                                      mask_2_hash_dict,
                                      model_list,
                                      PER_epsilon,
                                      PER_exponent,
                                      buffer_limit):

    buffer_limit_per_key = int( buffer_limit / len(list(present_state_tensor_dict.keys())) )
    
    for key in list(present_state_tensor_dict.keys()):

        history_state_tensor  = history_state_tensor_dict [key]
        history_action_tensor = history_action_tensor_dict[key]
        present_state_tensor  = present_state_tensor_dict [key]
        future_action_tensor  = future_action_tensor_dict [key]
        future_reward_tensor  = future_reward_tensor_dict [key]
        mask_1_tensor         = mask_1_tensor_dict        [key]
        mask_2_tensor         = mask_2_tensor_dict        [key]

        TD_error = 0
        for model in model_list:
            TD_error += obtain_TD_error(model, 
                                        history_state_tensor  ,
                                        history_action_tensor  ,
                                        present_state_tensor  ,
                                        future_action_tensor  ,
                                        future_reward_tensor  ,
                                        mask_1_tensor,
                                        mask_2_tensor   )
         
        TD_error             =(TD_error + PER_epsilon) ** PER_exponent
        TD_error_p           = TD_error / torch.sum(TD_error)
        indices              = torch.multinomial(TD_error_p, min(buffer_limit_per_key, len(TD_error_p)), replacement = False)

        history_state_tensor_dict [key] = history_state_tensor [indices]
        history_action_tensor_dict[key] = history_action_tensor[indices]
        present_state_tensor_dict [key] = present_state_tensor [indices]
        future_action_tensor_dict [key] = future_action_tensor [indices]
        future_reward_tensor_dict [key] = future_reward_tensor [indices]
        mask_1_tensor_dict        [key] = mask_1_tensor        [indices]
        mask_2_tensor_dict        [key] = mask_2_tensor        [indices]
        history_state_hash_dict   [key] = np.array(history_state_hash_dict [key])[indices.cpu().numpy()].tolist()
        history_action_hash_dict  [key] = np.array(history_action_hash_dict[key])[indices.cpu().numpy()].tolist()
        present_state_hash_dict   [key] = np.array(present_state_hash_dict [key])[indices.cpu().numpy()].tolist()
        future_action_hash_dict   [key] = np.array(future_action_hash_dict [key])[indices.cpu().numpy()].tolist()
        future_reward_hash_dict   [key] = np.array(future_reward_hash_dict [key])[indices.cpu().numpy()].tolist()
        mask_1_hash_dict          [key] = np.array(mask_1_hash_dict        [key])[indices.cpu().numpy()].tolist()
        mask_2_hash_dict          [key] = np.array(mask_2_hash_dict        [key])[indices.cpu().numpy()].tolist()

    return history_state_tensor_dict, history_action_tensor_dict, present_state_tensor_dict, future_action_tensor_dict, future_reward_tensor_dict, mask_1_tensor_dict, mask_2_tensor_dict,\
           history_state_hash_dict, history_action_hash_dict, present_state_hash_dict, future_action_hash_dict, future_reward_hash_dict, mask_1_hash_dict, mask_2_hash_dict




def update_model(iteration_for_learning,
                 history_state_tensor_dict,
                 history_action_tensor_dict,
                 present_state_tensor_dict,
                 future_action_tensor_dict,
                 future_reward_tensor_dict,
                 mask_1_tensor_dict ,
                 mask_2_tensor_dict ,
                 model,
                 PER_epsilon ,
                 PER_exponent,
                 device):

    for _ in range(iteration_for_learning):

        key                   = random.choice(list(present_state_tensor_dict.keys()))
        history_state_tensor  = history_state_tensor_dict [key]
        history_action_tensor = history_action_tensor_dict[key]
        present_state_tensor  = present_state_tensor_dict [key]
        future_action_tensor  = future_action_tensor_dict [key]
        future_reward_tensor  = future_reward_tensor_dict [key]
        mask_1_tensor         = mask_1_tensor_dict        [key]
        mask_2_tensor         = mask_2_tensor_dict        [key]

        """
        We update the TD error in the replay buffer after each training step (iteration) using the updated neural network.
        """
        TD_error             = obtain_TD_error (model, 
                                                history_state_tensor  ,
                                                history_action_tensor  ,
                                                present_state_tensor  ,
                                                future_action_tensor  ,
                                                future_reward_tensor  ,
                                                mask_1_tensor,
                                                mask_2_tensor   )

        TD_error             =(TD_error + PER_epsilon) ** PER_exponent
        TD_error_p           = TD_error / torch.sum(TD_error)
        indices              = torch.multinomial(TD_error_p, 1, replacement = True)

        history_state  = history_state_tensor [indices]
        history_action = history_action_tensor[indices]
        present_state  = present_state_tensor [indices]
        future_action  = future_action_tensor [indices]
        future_reward  = future_reward_tensor [indices]
        mask_1         = mask_1_tensor        [indices]
        mask_2         = mask_2_tensor        [indices]

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        output_reward               = model(history_state, history_action, present_state, future_action, (mask_1, mask_2))
        total_loss                  = loss_function(output_reward, future_reward)
        total_loss.backward()     # get grad

        selected_optimizer.step() # update params

    return model




def update_model_list(iteration_for_learning,
                      history_state_tensor_dict,
                      history_action_tensor_dict,
                      present_state_tensor_dict,
                      future_action_tensor_dict,
                      future_reward_tensor_dict,
                      mask_1_tensor_dict ,
                      mask_2_tensor_dict ,
                      model_list,  # List of models
                      PER_epsilon,
                      PER_exponent,
                      device):

    for i, model in enumerate(model_list):
        model_list[i] = update_model(iteration_for_learning,
                                     history_state_tensor_dict,
                                     history_action_tensor_dict,
                                     present_state_tensor_dict,
                                     future_action_tensor_dict,
                                     future_reward_tensor_dict,
                                     mask_1_tensor_dict ,
                                     mask_2_tensor_dict,
                                     model,
                                     PER_epsilon,
                                     PER_exponent,
                                     device)

    return model_list




def update_model_list_parallel(iteration_for_learning,
                               history_state_tensor_dict,
                               history_action_tensor_dict,
                               present_state_tensor_dict,
                               future_action_tensor_dict,
                               future_reward_tensor_dict,
                               mask_1_tensor_dict ,
                               mask_2_tensor_dict ,
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
                            history_state_tensor_dict,
                            history_action_tensor_dict,
                            present_state_tensor_dict,
                            future_action_tensor_dict,
                            future_reward_tensor_dict,
                            mask_1_tensor_dict ,
                            mask_2_tensor_dict,
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




def save_dicts_to_pickle(filename, *dicts):
    with open(filename, 'wb') as file:
        dill.dump(dicts, file)
