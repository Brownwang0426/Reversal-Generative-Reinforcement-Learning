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




def load_buffer_from_pickle(filename):
    with open(filename, 'rb') as file:
        list = dill.load(file)
    return list




def retrieve_history(state_list, action_list, history_size, device):
    if (len(action_list) >= 1) & (history_size != 0):
        history_state     = torch.stack(state_list  [-history_size-1:-1], dim=0).unsqueeze(0).to(device)
        history_action    = torch.stack(action_list [-history_size:]    , dim=0).unsqueeze(0).to(device)
    else:
        history_state     = torch.empty(0, 0, 0)
        history_action    = torch.empty(0, 0, 0)
    return history_state, history_action




def retrieve_present(state_list, device):
    return state_list[-1].unsqueeze(0).to(device)




def initialize_future_action(init, noise_t, noise_r, shape, device):
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




def initialize_desired_reward(shape, device):
    return  torch.ones(shape).to(device)




def update_future_action(iteration_for_deducing,
                         model_list,
                         history_state,
                         history_action,
                         present_state,
                         future_action,
                         desired_reward,
                         beta,
                         loss_scale):

    loss_weights = torch.tensor([loss_scale ** j for j in range(desired_reward.size(1))], device=desired_reward.device)

    for i in range(iteration_for_deducing):

        model              = random.choice(model_list)

        future_action_     = torch.sigmoid(future_action)
        future_action_     = future_action_.detach().requires_grad_(True)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()
        
        loss_function      = model.loss_function
        envisaged_reward, \
        envisaged_state    = model(history_state, history_action, present_state, future_action_)
        total_loss         = torch.sum(loss_function(envisaged_reward, desired_reward) * loss_weights)
        total_loss.backward() 

        future_action     -= future_action_.grad * (1 - future_action_) * future_action_ * beta 

    return future_action




def sequentialize(state_list, action_list, reward_list, sequence_size):

    present_state_list = []
    future_action_list = []
    future_reward_list = []
    future_state_list  = []

    for j in range(len(reward_list[:-sequence_size+1])):
        present_state_list.append(                  state_list [ j                           ]          )
        future_action_list.append(      torch.stack(action_list[ j   : j+sequence_size       ], dim=0)  )
        future_reward_list.append(      torch.stack(reward_list[ j   : j+sequence_size       ], dim=0)  )
        future_state_list.append (      torch.stack(state_list [ j+1 : j+sequence_size+1     ], dim=0)  )

    return present_state_list, future_action_list, future_reward_list, future_state_list




def hash_tensor(tensor):
    tensor = tensor.cpu()  
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

def fast_check_with_hash(hash_1d, hash_2d):
    return hash_1d not in hash_2d

def update_long_term_experience_replay_buffer(present_state_stack, 
                                              future_action_stack,
                                              future_reward_stack, 
                                              future_state_stack,
                                              present_state_hash_list, 
                                              future_action_hash_list, 
                                              future_reward_hash_list, 
                                              future_state_hash_list,
                                              present_state_list,
                                              future_action_list,
                                              future_reward_list,
                                              future_state_list ):

    for i in range(len(present_state_list)):
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

            present_state_stack = torch.cat((present_state_stack,    present_state.unsqueeze(0) ), dim=0)
            future_action_stack = torch.cat((future_action_stack,    future_action.unsqueeze(0) ), dim=0)
            future_reward_stack = torch.cat((future_reward_stack,    future_reward.unsqueeze(0) ), dim=0)
            future_state_stack  = torch.cat((future_state_stack,     future_state .unsqueeze(0) ), dim=0)
            present_state_hash_list.append( present_state_hash  )
            future_action_hash_list.append( future_action_hash  )
            future_reward_hash_list.append( future_reward_hash  )
            future_state_hash_list .append( future_state_hash   )

    return present_state_stack, future_action_stack, future_reward_stack, future_state_stack,\
           present_state_hash_list, future_action_hash_list, future_reward_hash_list, future_state_hash_list




def obtain_TD_error(model,
                    present_state_stack,
                    future_action_stack,
                    future_reward_stack,
                    future_state_stack
                    ):

    dataset      = TensorDataset(present_state_stack,
                                 future_action_stack,
                                 future_reward_stack,
                                 future_state_stack  )
    data_loader  = DataLoader(dataset, batch_size = len(dataset), shuffle=False)

    for present_state, future_action, future_reward, future_state in data_loader:

        model.eval()

        loss_function                 = model.loss_function_
        envisaged_reward, \
        envisaged_state               = model(torch.empty(0, 0, 0), torch.empty(0, 0, 0), present_state, future_action)
        total_loss_A                  = loss_function(envisaged_reward, future_reward) 
        total_loss_B                  = loss_function(envisaged_state, future_state)

        total_loss                    = 0
        total_loss                   += torch.sum(torch.abs(total_loss_A), dim=(1, 2))
        total_loss                    = torch.sum(torch.abs(total_loss_B), dim=(1, 2))

        TD_error                      = total_loss.detach()

    return TD_error




def clear_long_term_experience_replay_buffer(present_state_stack, 
                                             future_action_stack,
                                             future_reward_stack, 
                                             future_state_stack,
                                             present_state_hash_list, 
                                             future_action_hash_list, 
                                             future_reward_hash_list, 
                                             future_state_hash_list ,
                                             model_list,
                                             PER_epsilon,
                                             PER_exponent,
                                             buffer_limit):

    TD_error = 0
    for model in model_list:
        TD_error += obtain_TD_error(model, 
                                    present_state_stack ,
                                    future_action_stack ,
                                    future_reward_stack ,
                                    future_state_stack )
    
    TD_error             =(TD_error + PER_epsilon) ** PER_exponent
    TD_error_p           = TD_error / torch.sum(TD_error)
    indices              = torch.multinomial(TD_error_p, min(buffer_limit, len(TD_error_p)), replacement = False)

    present_state_stack  = present_state_stack  [indices]
    future_action_stack  = future_action_stack  [indices]
    future_reward_stack  = future_reward_stack  [indices]
    future_state_stack   = future_state_stack   [indices]
    present_state_hash_list = np.array(present_state_hash_list)[indices.cpu().numpy()].tolist()
    future_action_hash_list = np.array(future_action_hash_list)[indices.cpu().numpy()].tolist()
    future_reward_hash_list = np.array(future_reward_hash_list)[indices.cpu().numpy()].tolist()
    future_state_hash_list  = np.array(future_state_hash_list )[indices.cpu().numpy()].tolist()

    return present_state_stack, future_action_stack, future_reward_stack, future_state_stack,\
           present_state_hash_list, future_action_hash_list, future_reward_hash_list, future_state_hash_list




def update_model(iteration_for_learning,
                 present_state_stack,
                 future_action_stack,
                 future_reward_stack,
                 future_state_stack ,
                 model,
                 PER_epsilon ,
                 PER_exponent):

    TD_error             = obtain_TD_error (model, 
                                            present_state_stack  ,
                                            future_action_stack  ,
                                            future_reward_stack  ,
                                            future_state_stack )

    TD_error             =(TD_error + PER_epsilon) ** PER_exponent
    TD_error_p           = TD_error / torch.sum(TD_error)
    indices              = torch.multinomial(TD_error_p, iteration_for_learning, replacement = True)
    indices              = indices[torch.randperm(indices.size(0))]

    for i in range(len(indices)):

        present_state = present_state_stack[indices[i]].unsqueeze(0)
        future_action = future_action_stack[indices[i]].unsqueeze(0)
        future_reward = future_reward_stack[indices[i]].unsqueeze(0)
        future_state  = future_state_stack [indices[i]].unsqueeze(0)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        envisaged_reward, \
        envisaged_state             = model(torch.empty(0, 0, 0), torch.empty(0, 0, 0), present_state, future_action)
        total_loss                  = loss_function(envisaged_reward, future_reward) + loss_function(envisaged_state, future_state )
        total_loss.backward()     

        selected_optimizer.step() 

    return model




def update_model_list(iteration_for_learning,
                      present_state_stack,
                      future_action_stack,
                      future_reward_stack,
                      future_state_stack,
                      model_list, 
                      PER_epsilon,
                      PER_exponent):

    for i, model in enumerate(model_list):
        model_list[i] = update_model(iteration_for_learning,
                                     present_state_stack,
                                     future_action_stack,
                                     future_reward_stack,
                                     future_state_stack,
                                     model,
                                     PER_epsilon,
                                     PER_exponent)

    return model_list




def update_model_list_parallel(iteration_for_learning,
                               present_state_stack,
                               future_action_stack,
                               future_reward_stack,
                               future_state_stack ,
                               model_list, 
                               PER_epsilon,
                               PER_exponent):

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_model = {
            executor.submit(update_model, 
                            iteration_for_learning,
                            present_state_stack,
                            future_action_stack,
                            future_reward_stack,
                            future_state_stack,
                            model,
                            PER_epsilon,
                            PER_exponent): model
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




def save_buffer_to_pickle(filename, *list):
    with open(filename, 'wb') as file:
        dill.dump(list, file)
