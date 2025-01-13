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
                                present_state,
                                pre_activated_future_action,
                                desired_reward,
                                beta,
                                device):

    present_state, pre_activated_future_action, desired_reward = present_state.to(device), pre_activated_future_action.to(device), desired_reward.to(device)

    model_list_copy   = copy.deepcopy(model_list)

    time_size         = pre_activated_future_action.size(1)

    loss_scale        = 1.1

    for i in range(iteration_for_deducing):

        index         = np.random.randint(len(model_list_copy))
        model         = model_list_copy[index]
        tgt_indx      = np.random.randint(time_size) 

        future_action = torch.sigmoid(pre_activated_future_action[:, :tgt_indx+1])

        model.train()
        future_action = future_action.detach().requires_grad_(True)
        for param in model.parameters():
            param.requires_grad = False

        loss_function       = model.loss_function
        output_reward, _    = model(present_state, future_action)
        total_loss          = loss_function(output_reward[:, tgt_indx], desired_reward[:, tgt_indx])
        total_loss          = total_loss * (loss_scale ** tgt_indx)
        total_loss.backward() # get grad

        pre_activated_future_action[:, :tgt_indx+1] -= future_action.grad * (1 - future_action) * future_action * beta # update params
    
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
    
    for i in range(chunk_size):

        chunk_size_ = i + 1

        if chunk_size_ != 1:
            process_len = len(reward_list[:-chunk_size_+1])
        else:
            process_len = len(reward_list[:])

        for j in range(process_len):
            present_state_list.append(                  state_list [ j                         ]          .to(device)  )
            future_action_list.append(      torch.stack(action_list[ j   : j+chunk_size_       ], dim=0)  .to(device)  )
            future_reward_list.append(      torch.stack(reward_list[ j   : j+chunk_size_       ], dim=0)  .to(device)  )
            future_state_list.append(       torch.stack(state_list [ j+1 : j+chunk_size_+1     ], dim=0)  .to(device)  )

    return present_state_list, future_action_list, future_reward_list, future_state_list




def hash_tensor(tensor):
    tensor = tensor.cpu()  # Make sure the tensor is on the CPU
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

def fast_check_with_hash(hash_1d, hash_2d):
    return hash_1d not in hash_2d

def update_long_term_experience_buffer(present_state_tensor_dict, 
                                       future_action_tensor_dict,
                                       future_reward_tensor_dict, 
                                       future_state_tensor_dict,
                                       present_state_hash_dict, 
                                       future_action_hash_dict, 
                                       future_reward_hash_dict, 
                                       future_state_hash_dict,
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

        if  fast_check_with_hash(present_state_hash  , present_state_hash_dict[length]) or   \
            fast_check_with_hash(future_action_hash  , future_action_hash_dict[length]) or   \
            fast_check_with_hash(future_reward_hash  , future_reward_hash_dict[length]) or   \
            fast_check_with_hash(future_state_hash   , future_state_hash_dict [length]) :

            present_state_tensor_dict  [length] = torch.cat((present_state_tensor_dict  [length],    present_state.unsqueeze(0) ), dim=0)
            future_action_tensor_dict  [length] = torch.cat((future_action_tensor_dict  [length],    future_action.unsqueeze(0) ), dim=0)
            future_reward_tensor_dict  [length] = torch.cat((future_reward_tensor_dict  [length],    future_reward.unsqueeze(0) ), dim=0)
            future_state_tensor_dict   [length] = torch.cat((future_state_tensor_dict   [length],    future_state .unsqueeze(0) ), dim=0)
            present_state_hash_dict    [length] .append( present_state_hash  )
            future_action_hash_dict    [length] .append( future_action_hash  )
            future_reward_hash_dict    [length] .append( future_reward_hash  )
            future_state_hash_dict     [length] .append( future_state_hash   )

    return present_state_tensor_dict, future_action_tensor_dict, future_reward_tensor_dict, future_state_tensor_dict,\
           present_state_hash_dict, future_action_hash_dict, future_reward_hash_dict, future_state_hash_dict




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

        """
        We strongely suggest you not to use total_loss_B because:
        1 - The orignal meaning in PER is that the suprising experiences are taken into account with priority.
        2 - The meaning of "surpising" mainly points to how the predicted reward deviates from actual reward, not states.
        3 - In our experiments, taking states into account in PER does jeopardize the performance.
        """
        loss_function                 = model.loss_function_
        output_reward, output_state   = model(present_state, future_action)
        total_loss_A                  = loss_function(output_reward[:, -1], future_reward[:, -1]) 
        # total_loss_B                  = loss_function(output_state, future_state)

        total_loss                    = 0
        total_loss                   += torch.sum(torch.abs(total_loss_A), dim=(1))
        # total_loss                   += torch.sum(torch.abs(total_loss_B), dim=(1, 2))

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


    for _ in range(iteration_for_learning):

        key                  = random.choice(list(present_state_tensor_dict.keys()))
        present_state_tensor = present_state_tensor_dict[key]
        future_action_tensor = future_action_tensor_dict[key]
        future_reward_tensor = future_reward_tensor_dict[key]
        future_state_tensor  = future_state_tensor_dict [key]

        """
        We update the TD error in the replay buffer after each training step (iteration) using the updated neural network.
        """
        TD_error             = obtain_TD_error (model, 
                                                present_state_tensor  ,
                                                future_action_tensor  ,
                                                future_reward_tensor  ,
                                                future_state_tensor   )

        """
        TD error clipping is a common practice in PER to prevent the model from being overwhelmed by outliers.
        """
        TD_error             = torch.clamp(TD_error, min=0, max=1e20)  
        TD_error             =(TD_error + PER_epsilon) ** PER_exponent
        TD_error             = torch.clamp(TD_error, min=0, max=1e20)  
        TD_error_p           = TD_error / (torch.sum(TD_error) + 1e-20)
        TD_error_p           = torch.clamp(TD_error_p, min=0, max=1) 
        index                = torch.multinomial(TD_error_p, 1, replacement = True)[0]

        present_state = present_state_tensor [index].unsqueeze(0)
        future_action = future_action_tensor [index].unsqueeze(0)
        future_reward = future_reward_tensor [index].unsqueeze(0)
        future_state  = future_state_tensor  [index].unsqueeze(0)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        output_reward, output_state = model(present_state, future_action)
        total_loss                  = loss_function(output_reward[:, -1], future_reward[:, -1]) + loss_function(output_state, future_state )
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




def clear_long_term_experience_buffer(present_state_tensor_dict, 
                                      future_action_tensor_dict,
                                      future_reward_tensor_dict, 
                                      future_state_tensor_dict,
                                      present_state_hash_dict, 
                                      future_action_hash_dict, 
                                      future_reward_hash_dict, 
                                      future_state_hash_dict ,
                                      model_list,
                                      PER_epsilon,
                                      PER_exponent,
                                      buffer_limit):
    
    for key in list(present_state_tensor_dict.keys()):

        present_state_tensor = present_state_tensor_dict[key]
        future_action_tensor = future_action_tensor_dict[key]
        future_reward_tensor = future_reward_tensor_dict[key]
        future_state_tensor  = future_state_tensor_dict [key]

        TD_error = 0
        for model in model_list:
            TD_error += obtain_TD_error(model, 
                                        present_state_tensor  ,
                                        future_action_tensor  ,
                                        future_reward_tensor  ,
                                        future_state_tensor   )
        
        """
        TD error clipping is a common practice in PER to prevent the model from being overwhelmed by outliers.
        """
        TD_error             = torch.clamp(TD_error, min=0, max=1e20)  
        TD_error             =(TD_error + PER_epsilon) ** PER_exponent
        TD_error             = torch.clamp(TD_error, min=0, max=1e20)  
        TD_error_p           = TD_error / (torch.sum(TD_error) + 1e-20)
        TD_error_p           = torch.clamp(TD_error_p, min=0, max=1) 
        indices              = torch.multinomial(TD_error_p, min(buffer_limit, len(TD_error_p)), replacement = False)

        present_state_tensor_dict [key] = present_state_tensor_dict [key][indices]
        future_action_tensor_dict [key] = future_action_tensor_dict [key][indices]
        future_reward_tensor_dict [key] = future_reward_tensor_dict [key][indices]
        future_state_tensor_dict  [key] = future_state_tensor_dict  [key][indices]
        present_state_hash_dict   [key] = [present_state_hash_dict  [key][i] for i in indices.tolist()]
        future_action_hash_dict   [key] = [future_action_hash_dict  [key][i] for i in indices.tolist()]
        future_reward_hash_dict   [key] = [future_reward_hash_dict  [key][i] for i in indices.tolist()]
        future_state_hash_dict    [key] = [future_state_hash_dict   [key][i] for i in indices.tolist()]

    return present_state_tensor_dict, future_action_tensor_dict, future_reward_tensor_dict, future_state_tensor_dict,\
           present_state_hash_dict, future_action_hash_dict, future_reward_hash_dict, future_state_hash_dict




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)

