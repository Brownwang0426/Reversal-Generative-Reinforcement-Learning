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
    if history_size != 0:
        history_state     = torch.stack(state_list  [-history_size-1:-1], dim=0).unsqueeze(0).to(device)
        history_action    = torch.stack(action_list [-history_size:]    , dim=0).unsqueeze(0).to(device)
    else:
        history_state     = torch.empty(0, 0, 0).to(device)
        history_action    = torch.empty(0, 0, 0).to(device)
    return history_state, history_action




def retrieve_present(state_list, device):
    return state_list[-1].unsqueeze(0).to(device)




def initialize_future_action(init, noise_t, noise_r, shape, device):
    input = 0
    if   init == "random_uniform":
        for _ in range(noise_t):
            input += (( 2 * torch.rand  (shape)                     -1 ) * noise_r).to(device) 
    elif init == "random_normal":
        for _ in range(noise_t):
            input += ((     torch.normal(mean=0, std=1, size=shape)    ) * noise_r).to(device) 
    return input




def initialize_desired_reward(shape, device):
    return  torch.ones(shape).to(device)




def update_future_action(itrtn_for_planning,
                         model_list,
                         history_state,
                         history_action,
                         present_state,
                         future_action,
                         desired_reward,
                         beta):
    
    desired_reward = desired_reward[:, -1, :]

    for _ in range(itrtn_for_planning):

        model              = random.choice(model_list)

        future_action_     = torch.tanh(future_action)
        future_action_     = future_action_.detach().requires_grad_(True)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()
        
        loss_function      = model.loss_function
        envisaged_reward, \
        envisaged_state    = model(history_state, history_action, present_state, future_action_)
        total_loss         = loss_function(envisaged_reward[:, -1, :], desired_reward)
        total_loss.backward() 

        future_action     -= future_action_.grad * (1 - future_action_ * future_action_) * beta 

    return future_action




def sequentialize(state_list, action_list, reward_list, history_size, future_size):

    device              = state_list[0].device
    torch_empty         = torch.empty(0, 0, 0).to(device)

    history_state_list  = []
    history_action_list = []
    present_state_list  = []
    future_action_list  = []
    future_reward_list  = []
    future_state_list   = []

    if history_size > 0:

        for i in range(len(reward_list[:-history_size-future_size+1])):

            history_state_list.append (      torch.stack(state_list [ i                     : i + history_size                     ], dim=0)  )
            history_action_list.append(      torch.stack(action_list[ i                     : i + history_size                     ], dim=0)  )
            present_state_list.append (                  state_list [ i + history_size                                             ]          )
            future_action_list.append (      torch.stack(action_list[ i + history_size      : i + history_size + future_size       ], dim=0)  )
            future_reward_list.append (      torch.stack(reward_list[ i + history_size      : i + history_size + future_size       ], dim=0)  )
            future_state_list.append  (      torch.stack(state_list [ i + history_size + 1  : i + history_size + future_size + 1   ], dim=0)  )

    else:

        for i in range(len(reward_list[:-history_size-future_size+1])):

            history_state_list.append (                  torch_empty                                                                          )
            history_action_list.append(                  torch_empty                                                                          )
            present_state_list.append (                  state_list [ i + history_size                                             ]          )
            future_action_list.append (      torch.stack(action_list[ i + history_size      : i + history_size + future_size       ], dim=0)  )
            future_reward_list.append (      torch.stack(reward_list[ i + history_size      : i + history_size + future_size       ], dim=0)  )
            future_state_list.append  (      torch.stack(state_list [ i + history_size + 1  : i + history_size + future_size + 1   ], dim=0)  )

    return history_state_list, history_action_list, present_state_list, future_action_list, future_reward_list, future_state_list




def hash_tensor(tensor):
    tensor = tensor.cpu()  
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

def fast_check_with_hash(hash_1d, hash_2d):
    return hash_1d not in hash_2d

def update_long_term_experience_replay_buffer(history_state_stack, 
                                              history_action_stack,
                                              present_state_stack, 
                                              future_action_stack,
                                              future_reward_stack, 
                                              future_state_stack,
                                              history_state_hash_list, 
                                              history_action_hash_list, 
                                              present_state_hash_list, 
                                              future_action_hash_list, 
                                              future_reward_hash_list, 
                                              future_state_hash_list,
                                              history_state_list,
                                              history_action_list,
                                              present_state_list,
                                              future_action_list,
                                              future_reward_list,
                                              future_state_list):

    for i in range(len(present_state_list)):
        history_state       = history_state_list  [i]
        history_action      = history_action_list [i]
        present_state       = present_state_list  [i]
        future_action       = future_action_list  [i]
        future_reward       = future_reward_list  [i]
        future_state        = future_state_list   [i]
        history_state_hash  = hash_tensor(history_state )
        history_action_hash = hash_tensor(history_action)
        present_state_hash  = hash_tensor(present_state )
        future_action_hash  = hash_tensor(future_action )
        future_reward_hash  = hash_tensor(future_reward )
        future_state_hash   = hash_tensor(future_state  )

        if  fast_check_with_hash(history_state_hash  , history_state_hash_list ) or   \
            fast_check_with_hash(history_action_hash , history_action_hash_list) or   \
            fast_check_with_hash(present_state_hash  , present_state_hash_list ) or   \
            fast_check_with_hash(future_action_hash  , future_action_hash_list ) or   \
            fast_check_with_hash(future_reward_hash  , future_reward_hash_list ) or   \
            fast_check_with_hash(future_state_hash   , future_state_hash_list  ) :

            history_state_stack     = torch.cat((history_state_stack,    history_state.unsqueeze (0) ), dim=0)
            history_action_stack    = torch.cat((history_action_stack,   history_action.unsqueeze(0) ), dim=0)
            present_state_stack     = torch.cat((present_state_stack,    present_state.unsqueeze (0) ), dim=0)
            future_action_stack     = torch.cat((future_action_stack,    future_action.unsqueeze (0) ), dim=0)
            future_reward_stack     = torch.cat((future_reward_stack,    future_reward.unsqueeze (0) ), dim=0)
            future_state_stack      = torch.cat((future_state_stack,     future_state .unsqueeze (0) ), dim=0)
            history_state_hash_list .append ( history_state_hash  )
            history_action_hash_list.append ( history_action_hash )
            present_state_hash_list .append ( present_state_hash  )
            future_action_hash_list .append ( future_action_hash  )
            future_reward_hash_list .append ( future_reward_hash  )
            future_state_hash_list  .append ( future_state_hash   )
        
        else:
            pass
        
    return history_state_stack, history_action_stack, present_state_stack, future_action_stack, future_reward_stack, future_state_stack,\
           history_state_hash_list, history_action_hash_list, present_state_hash_list, future_action_hash_list, future_reward_hash_list, future_state_hash_list




def find_optimal_batch_size(model, dataset, device='cuda:0', bs_list=None, max_mem_ratio=0.9):
    import torch, time, gc
    from torch.utils.data import DataLoader

    if bs_list is None:
        bs_list = [32, 64, 128, 256, 512, 1024]

    torch.cuda.set_device(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    results = []

    for bs in bs_list:
        torch.cuda.empty_cache(); gc.collect()
        loader = DataLoader(dataset, batch_size=bs, shuffle=False)
        batch = next(iter(loader))
        try:
            batch = [x.to(device) for x in batch]
            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            start = time.time()
            with torch.no_grad():
                model(*batch[:4])  
            duration = time.time() - start
            peak_mem = torch.cuda.max_memory_allocated(device)
            mem_ratio = peak_mem / total_mem

            if mem_ratio < max_mem_ratio:
                results.append((bs, duration, mem_ratio))
                # print(f"✅ bs={bs:4d} 🕒 {duration:.3f}s  💾 {mem_ratio:.1%}")
                pass
            else:
                # print(f"❌ bs={bs:4d} exceeded memory limit: {mem_ratio:.1%}")
                pass
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # print(f"❌ bs={bs:4d} triggerd OOM")
                pass
            else:
                raise e

    if not results:
        raise RuntimeError("all batch size OOM")

    best_bs = min(results, key=lambda x: x[1])[0]
    # print(f"\n✅ best batch size：{best_bs}")
    return best_bs

def obtain_obsolute_TD_error(model,
                             history_state_stack,
                             history_action_stack,
                             present_state_stack,
                             future_action_stack,
                             future_reward_stack,
                             future_state_stack,
                             batch_size
                             ):
    
    dataset      = TensorDataset(history_state_stack,
                                 history_action_stack,
                                 present_state_stack,
                                 future_action_stack,
                                 future_reward_stack,
                                 future_state_stack  )
    data_loader  = DataLoader(dataset, batch_size = batch_size, shuffle=False)
    
    TD_error_list = []

    model.eval()
    loss_function = model.loss_function_
    with torch.no_grad():
        for history_state, history_action, present_state, future_action, future_reward, future_state in data_loader:
            envisaged_reward, \
            envisaged_state               = model(history_state, history_action, present_state, future_action)
            total_loss_reward             = loss_function(envisaged_reward, future_reward) 
            total_loss_state              = loss_function(envisaged_state, future_state) 
            total_loss                    = torch.sum(torch.abs(total_loss_reward), dim=(1, 2)) + torch.sum(torch.abs(total_loss_state), dim=(1, 2)) 
            TD_error_list.append(total_loss)
    
    TD_error = torch.cat(TD_error_list)

    return TD_error

def update_model(itrtn_for_learning,
                 history_state_stack,
                 history_action_stack,
                 present_state_stack,
                 future_action_stack,
                 future_reward_stack,
                 future_state_stack ,
                 model,
                 batch_size_for_td_error,
                 batch_size_for_learning):

    PER_epsilon  = 1e-10
    PER_exponent = 2
    batch_size_for_learning = min(batch_size_for_learning, len(present_state_stack))

    for _ in tqdm(range(itrtn_for_learning)):
    
        obsolute_TD_error    = obtain_obsolute_TD_error(model, 
                                                        history_state_stack  ,
                                                        history_action_stack ,
                                                        present_state_stack  ,
                                                        future_action_stack  ,
                                                        future_reward_stack  ,
                                                        future_state_stack,
                                                        batch_size_for_td_error)
        priority             = obsolute_TD_error + PER_epsilon
        exponent_priority    = priority ** PER_exponent
        priority_probability = exponent_priority / torch.sum(exponent_priority)

        final_indice   = torch.multinomial(priority_probability, batch_size_for_learning, replacement=False)

        history_state  = history_state_stack [final_indice]
        history_action = history_action_stack[final_indice]
        present_state  = present_state_stack [final_indice]
        future_action  = future_action_stack [final_indice]
        future_reward  = future_reward_stack [final_indice]
        future_state   = future_state_stack  [final_indice]

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        envisaged_reward, \
        envisaged_state             = model(history_state, history_action, present_state, future_action)
        total_loss                  = loss_function(envisaged_reward, future_reward) + loss_function(envisaged_state, future_state )
        total_loss.backward()     

        selected_optimizer.step() 

    return model

def update_model_list(epoch_itrtn_for_learning,
                      history_state_stack,
                      history_action_stack,
                      present_state_stack,
                      future_action_stack,
                      future_reward_stack,
                      future_state_stack,
                      model_list,
                      batch_size_for_td_error,
                      batch_size_for_learning):

    for i, model in enumerate(model_list):
        model_list[i] = update_model(epoch_itrtn_for_learning,
                                     history_state_stack,
                                     history_action_stack,
                                     present_state_stack,
                                     future_action_stack,
                                     future_reward_stack,
                                     future_state_stack,
                                     model,
                                     batch_size_for_td_error,
                                     batch_size_for_learning)

    return model_list




def limit_buffer(history_state_stack, 
                 history_action_stack,
                 present_state_stack, 
                 future_action_stack,
                 future_reward_stack, 
                 future_state_stack,
                 history_state_hash_list, 
                 history_action_hash_list, 
                 present_state_hash_list, 
                 future_action_hash_list, 
                 future_reward_hash_list, 
                 future_state_hash_list,
                 buffer_limit ):

    probability              = torch.ones(len(present_state_stack)) / len(present_state_stack) 
    indices_to_keep          = torch.multinomial(probability, min(buffer_limit, len(present_state_stack)), replacement = False)

    history_state_stack      = history_state_stack      [indices_to_keep]
    history_action_stack     = history_action_stack     [indices_to_keep]
    present_state_stack      = present_state_stack      [indices_to_keep]
    future_action_stack      = future_action_stack      [indices_to_keep]
    future_reward_stack      = future_reward_stack      [indices_to_keep]
    future_state_stack       = future_state_stack       [indices_to_keep]
    history_state_hash_list  = [history_state_hash_list [i] for i in indices_to_keep]
    history_action_hash_list = [history_action_hash_list[i] for i in indices_to_keep]
    present_state_hash_list  = [present_state_hash_list [i] for i in indices_to_keep]
    future_action_hash_list  = [future_action_hash_list [i] for i in indices_to_keep]
    future_reward_hash_list  = [future_reward_hash_list [i] for i in indices_to_keep]
    future_state_hash_list   = [future_state_hash_list  [i] for i in indices_to_keep]
        
    return history_state_stack, history_action_stack, present_state_stack, future_action_stack, future_reward_stack, future_state_stack,\
           history_state_hash_list, history_action_hash_list, present_state_hash_list, future_action_hash_list, future_reward_hash_list, future_state_hash_list




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)




def save_buffer_to_pickle(filename, *list):
    with open(filename, 'wb') as file:
        dill.dump(list, file)
