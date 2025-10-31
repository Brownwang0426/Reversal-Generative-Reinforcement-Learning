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
from tqdm.auto import tqdm
from collections import defaultdict

import itertools

import dill

import warnings
warnings.filterwarnings('ignore')

import concurrent.futures
import hashlib

import torch, time, gc
from torch.utils.data import DataLoader

import zlib


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
        history_state     = torch.stack(state_list  [-history_size-1:-1], dim=0).unsqueeze(0).to(device, non_blocking=True)
    else:
        history_state     = torch.empty(0, 0, 0).to(device, non_blocking=True)
    return history_state




def retrieve_present(state_list, device):
    return state_list[-1].unsqueeze(0).to(device, non_blocking=True)




def initialize_future_action(shape, device):
    return torch.zeros(shape).to(device, non_blocking=True)




def initialize_desired_reward(shape, device):
    return  torch.ones(shape).to(device, non_blocking=True)




def update_future_action(itrtn_for_planning,
                         model_list,
                         history_state,
                         present_state,
                         future_action,
                         desired_reward,
                         beta):

    device = next(model_list[0].parameters()).device
    device_ = history_state.device

    history_state  = history_state.to(device)
    present_state  = present_state.to(device)
    future_action  = future_action.to(device)
    desired_reward = desired_reward.to(device)

    for _ in range(itrtn_for_planning):

        model              = random.choice(model_list)

        future_action_     = torch.tanh(future_action)
        future_action_     = future_action_.detach().requires_grad_(True)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()
        
        loss_function      = model.loss_function
        envisaged_reward   = model(history_state, present_state, future_action_)
        total_loss         = loss_function(envisaged_reward[:, -1, :], desired_reward[:, -1, :])
        total_loss.backward() 

        future_action     -= future_action_.grad * (1 - future_action_ * future_action_) * beta 

    future_action = future_action.to(device_, non_blocking=True)

    return future_action




def sequentialize(state_list, action_list, reward_list, history_size, future_size):

    device              = state_list[0].device
    torch_empty         = torch.empty(0, 0, 0).to(device, non_blocking=True)

    history_state_list  = []
    present_state_list  = []
    future_action_list  = []
    future_reward_list  = []

    if history_size > 0:

        for i in range(len(reward_list[:-history_size-future_size+1])):

            history_state_list.append (      torch.stack(state_list [ i                     : i + history_size                     ], dim=0)  )
            present_state_list.append (                  state_list [ i + history_size                                             ]          )
            future_action_list.append (      torch.stack(action_list[ i + history_size      : i + history_size + future_size       ], dim=0)  )
            future_reward_list.append (      torch.stack(reward_list[ i + history_size      : i + history_size + future_size       ], dim=0)  )

    else:

        for i in range(len(reward_list[:-history_size-future_size+1])):

            history_state_list.append (                  torch_empty                                                                          )
            present_state_list.append (                  state_list [ i + history_size                                             ]          )
            future_action_list.append (      torch.stack(action_list[ i + history_size      : i + history_size + future_size       ], dim=0)  )
            future_reward_list.append (      torch.stack(reward_list[ i + history_size      : i + history_size + future_size       ], dim=0)  )

    return history_state_list, present_state_list, future_action_list, future_reward_list




def fast_hash_tensor(tensor):
    arr = tensor.detach().cpu().view(-1)
    sample = arr.numpy().tobytes()
    return zlib.adler32(sample)

def update_long_term_experience_replay_buffer(
        history_state_stack, present_state_stack, future_action_stack, future_reward_stack,
        history_state_hash_set, present_state_hash_set, future_action_hash_set, future_reward_hash_set,
        history_state_list, present_state_list, future_action_list, future_reward_list):

    new_history_state_list, new_present_state_list, new_future_action_list, new_future_reward_list = [], [], [], []

    for i in range(len(present_state_list)):
        history_state = history_state_list[i]
        present_state = present_state_list[i]
        future_action = future_action_list[i]
        future_reward = future_reward_list[i]

        h_hash = fast_hash_tensor(history_state)
        p_hash = fast_hash_tensor(present_state)
        a_hash = fast_hash_tensor(future_action)
        r_hash = fast_hash_tensor(future_reward)

        if (h_hash not in history_state_hash_set or
            p_hash not in present_state_hash_set or
            a_hash not in future_action_hash_set or
            r_hash not in future_reward_hash_set):

            new_history_state_list.append(history_state.unsqueeze(0))
            new_present_state_list.append(present_state.unsqueeze(0))
            new_future_action_list.append(future_action.unsqueeze(0))
            new_future_reward_list.append(future_reward.unsqueeze(0))

            history_state_hash_set.add(h_hash)
            present_state_hash_set.add(p_hash)
            future_action_hash_set.add(a_hash)
            future_reward_hash_set.add(r_hash)

    if new_present_state_list:
        history_state_stack = torch.cat([history_state_stack] + new_history_state_list, dim=0)
        present_state_stack = torch.cat([present_state_stack] + new_present_state_list, dim=0)
        future_action_stack = torch.cat([future_action_stack] + new_future_action_list, dim=0)
        future_reward_stack = torch.cat([future_reward_stack] + new_future_reward_list, dim=0)

    return (history_state_stack, present_state_stack, future_action_stack, future_reward_stack,
            history_state_hash_set, present_state_hash_set, future_action_hash_set, future_reward_hash_set)




def update_model(itrtn_for_learning,
                 dataset,
                 model,
                 batch_size):
    
    device = next(model.parameters()).device

    for _ in range(itrtn_for_learning):

        random_indices = random.sample(range(len(dataset)), batch_size)

        batch_samples  = [dataset[i] for i in random_indices]
        history_state, present_state, future_action, future_reward = zip(*batch_samples)
        history_state  = torch.stack(history_state ).to(device)
        present_state  = torch.stack(present_state ).to(device)
        future_action  = torch.stack(future_action ).to(device)
        future_reward  = torch.stack(future_reward ).to(device)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        envisaged_reward            = model(history_state, present_state, future_action)
        total_loss                  = loss_function(envisaged_reward[:, -1, :], future_reward[:, -1, :]) 
        total_loss.backward()     

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        selected_optimizer.step() 

    return model

def update_model_list(itrtn_for_learning,
                      dataset,
                      model_list,
                      batch_size):
    for i, model in enumerate(tqdm(model_list, desc="Updating models")):
        model_list[i] = update_model(itrtn_for_learning,
                                     dataset,
                                     model,
                                     batch_size)
    return model_list




def limit_buffer(history_state_stack, 
                 present_state_stack, 
                 future_action_stack,
                 future_reward_stack,
                 history_state_hash_set, 
                 present_state_hash_set, 
                 future_action_hash_set, 
                 future_reward_hash_set, 
                 buffer_limit):

    n = len(present_state_stack)
    probability = torch.ones(n) / n
    indices_to_keep = torch.multinomial(probability, min(buffer_limit, n), replacement=False).tolist()

    # slice tensor buffers
    history_state_stack = history_state_stack[indices_to_keep]
    present_state_stack = present_state_stack[indices_to_keep]
    future_action_stack = future_action_stack[indices_to_keep]
    future_reward_stack = future_reward_stack[indices_to_keep]

    h_hash_set = set()
    p_hash_set = set()
    a_hash_set = set()
    r_hash_set = set()
    for i in range(len(history_state_stack)):
        h_hash_set.add(fast_hash_tensor(history_state_stack[i]))
        p_hash_set.add(fast_hash_tensor(present_state_stack[i]))
        a_hash_set.add(fast_hash_tensor(future_action_stack[i]))
        r_hash_set.add(fast_hash_tensor(future_reward_stack[i]))

    history_state_hash_set = history_state_hash_set & h_hash_set
    present_state_hash_set = present_state_hash_set & p_hash_set
    future_action_hash_set = future_action_hash_set & a_hash_set
    future_reward_hash_set = future_reward_hash_set & r_hash_set

    return (history_state_stack, present_state_stack, future_action_stack, future_reward_stack,
            history_state_hash_set, present_state_hash_set, future_action_hash_set, future_reward_hash_set)




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)




def save_buffer_to_pickle(filename, *list):
    with open(filename, 'wb') as file:
        dill.dump(list, file)




# experimental --------------------------------------------------------------------

def _update_model_per_(itrtn_for_learning,
                       dataset,
                       model,
                       optimal_batch_size):
    
    device         = next(model.parameters()).device
    td_error_batch = optimal_batch_size
    PER_epsilon    = 1e-10
    PER_exponent   = 0.5

    for _ in tqdm(range(itrtn_for_learning)):

        batch_samples  = [dataset[0]]
        history_state, present_state, future_action, future_reward = zip(*batch_samples)
        history_state  = torch.stack(history_state ).to(device)
        present_state  = torch.stack(present_state ).to(device)
        future_action  = torch.stack(future_action ).to(device)
        future_reward  = torch.stack(future_reward ).to(device)
        model.train()
        _ = model(history_state, present_state, future_action)
        model.lock()

        obsolute_TD_error    = obtain_obsolute_TD_error(model, dataset, td_error_batch, device)
        priority             = obsolute_TD_error + PER_epsilon
        exponent_priority    = priority ** PER_exponent
        priority_probability = exponent_priority / torch.sum(exponent_priority)
        final_indices        = torch.multinomial(priority_probability, 1, replacement=False)

        batch_samples  = [dataset[i] for i in final_indices]
        history_state, present_state, future_action, future_reward = zip(*batch_samples)
        history_state  = torch.stack(history_state ).to(device)
        present_state  = torch.stack(present_state ).to(device)
        future_action  = torch.stack(future_action ).to(device)
        future_reward  = torch.stack(future_reward ).to(device)

        model.train()
        model.lock()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        envisaged_reward            = model(history_state, present_state, future_action)
        total_loss                  = loss_function(envisaged_reward, future_reward) 
        total_loss.backward()     

        selected_optimizer.step() 

    return model

def find_optimal_batch_size(model, dataset, device='cuda:0', bs_list=None, max_mem_ratio=0.9):

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
            hs, ps, fa, fr = batch
            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            start = time.time()
            with torch.no_grad():
                model.forward(hs, ps, fa)  
            duration = time.time() - start
            peak_mem = torch.cuda.max_memory_allocated(device)
            mem_ratio = peak_mem / total_mem

            if mem_ratio < max_mem_ratio:
                results.append((bs, duration, mem_ratio))
                pass
            else:
                pass
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pass
            else:
                raise e

    if not results:
        raise RuntimeError("all batch size OOM")

    best_bs = min(results, key=lambda x: x[1])[0]
    return best_bs

def obtain_obsolute_TD_error(model, dataset, td_error_batch, device):

    data_loader  = DataLoader(dataset, batch_size = td_error_batch, shuffle=False, pin_memory=True, num_workers=0)
    
    TD_error_list = []

    for history_state, present_state, future_action, future_reward in data_loader:

        history_state  = history_state.to(device)
        present_state  = present_state.to(device)
        future_action  = future_action.to(device)
        future_reward  = future_reward.to(device)

        model.train()
        model.lock()

        loss_function                 = model.loss_function_
        envisaged_reward              = model(history_state, present_state, future_action)
        total_loss                    = torch.sum(torch.abs(loss_function(envisaged_reward, future_reward) ), dim=(1, 2))
        TD_error_list.append(total_loss.detach())  

    TD_error = torch.cat(TD_error_list, dim=0).to(device)

    return TD_error

def update_model_per(itrtn_for_learning,
                     dataset,
                     model,
                     optimal_batch_size):
    
    device         = next(model.parameters()).device
    td_error_batch = optimal_batch_size
    PER_epsilon    = 1e-10
    PER_exponent   = 0.5

    batch_samples  = [dataset[0]]
    history_state, present_state, future_action, future_reward = zip(*batch_samples)
    history_state  = torch.stack(history_state ).to(device)
    present_state  = torch.stack(present_state ).to(device)
    future_action  = torch.stack(future_action ).to(device)
    future_reward  = torch.stack(future_reward ).to(device)
    model.train()
    model.unlock()
    _ = model(history_state, present_state, future_action)
    model.lock()

    obsolute_TD_error    = obtain_obsolute_TD_error(model, dataset, td_error_batch, device)

    for _ in tqdm(range(itrtn_for_learning)):

        priority             = obsolute_TD_error + PER_epsilon
        exponent_priority    = priority ** PER_exponent
        priority_probability = exponent_priority / torch.sum(exponent_priority)
        final_indices        = torch.multinomial(priority_probability, 1, replacement=False)

        batch_samples  = [dataset[i] for i in final_indices.cpu().tolist()]
        history_state, present_state, future_action, future_reward = zip(*batch_samples)
        history_state  = torch.stack(history_state ).to(device)
        present_state  = torch.stack(present_state ).to(device)
        future_action  = torch.stack(future_action ).to(device)
        future_reward  = torch.stack(future_reward ).to(device)

        model.train()
        model.lock()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()
        loss_function               = model.loss_function
        envisaged_reward            = model(history_state, present_state, future_action)
        total_loss                  = loss_function(envisaged_reward, future_reward) 
        total_loss.backward()     
        selected_optimizer.step() 

        model.train()
        model.lock()
        loss_function               = model.loss_function_
        envisaged_reward            = model(history_state, present_state, future_action)
        total_loss                  = torch.sum(torch.abs(loss_function(envisaged_reward, future_reward) ), dim=(1, 2))
        obsolute_TD_error[final_indices] =  total_loss.detach()

    return model

def _update_model_list_(itrtn_for_learning,
                      dataset,
                      model_list,
                      per=False):
    if per:
        device = next(model_list[0].parameters()).device
        optimal_batch_size = find_optimal_batch_size(model_list[0], dataset, device=device)
        for i, model in enumerate(model_list):
            model_list[i] = update_model_per(itrtn_for_learning,
                                             dataset,
                                             model,
                                             optimal_batch_size)
    else:
        for i, model in enumerate(model_list):
            model_list[i] = update_model(itrtn_for_learning,
                                         dataset,
                                         model)
    return model_list