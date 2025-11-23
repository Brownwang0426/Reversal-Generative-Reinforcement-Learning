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
        history_action    = torch.stack(action_list [-history_size:]    , dim=0).unsqueeze(0).to(device, non_blocking=True)
    else:
        history_state     = torch.empty(0, 0, 0).to(device, non_blocking=True)
        history_action    = torch.empty(0, 0, 0).to(device, non_blocking=True)
    return history_state, history_action



def retrieve_present(state_list, device):
    return state_list[-1].unsqueeze(0).to(device, non_blocking=True)




def initialize_future_action(shape, device):
    return torch.zeros(shape).to(device, non_blocking=True)




def initialize_desired_reward(shape, device):
    return torch.ones(shape).to(device, non_blocking=True)




def update_future_action(itrtn_for_planning,
                         model_list,
                         history_state,
                         history_action,
                         present_state,
                         future_action,
                         desired_reward,
                         beta):

    device = next(model_list[0].parameters()).device
    device_ = history_state.device

    history_state  = history_state.to(device)
    history_action = history_action.to(device)
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
        envisaged_reward, \
        envisaged_state    = model._forward(history_state, history_action, present_state, None, future_action_)
        total_loss         = loss_function(envisaged_reward[:, -1:, :], desired_reward[:, -1:, :])
        total_loss.backward() 

        grad = future_action_.grad
        grad = grad * (1 - future_action_ * future_action_)

        # ----- custom gradient update -----

        grad_sign   = grad.sign()
        grad_abs    = grad.abs()
        base        = torch.tanh(2 * grad_abs)
        decay       = torch.exp(-(grad_abs.mean() - 1).clamp(min=0))
        scaled_grad = grad_sign * base * decay

        # grad_sign   = grad.sign()
        # grad_abs    = grad.abs()
        # base        = torch.tanh(3 * grad_abs)
        # decay       = torch.exp(-(grad_abs - 1).clamp(min=0))
        # scaled_grad = grad_sign * base * decay 
        
        # ----- custom gradient update -----

        future_action = future_action - beta * scaled_grad

    future_action = future_action.to(device_, non_blocking=True)

    return future_action




def sequentialize(state_list, action_list, reward_list, history_size, future_size):

    device              = state_list[0].device
    torch_empty         = torch.empty(0, 0, 0).to(device, non_blocking=True)

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




def fast_hash_tensor(tensor):
    arr = tensor.detach().cpu().view(-1)
    sample = arr.numpy().tobytes()
    return zlib.adler32(sample)

def update_long_term_experience_replay_buffer(history_state_stack, 
                                              history_action_stack,
                                              present_state_stack, 
                                              future_action_stack,
                                              future_reward_stack, 
                                              future_state_stack,
                                              history_state_hash_set , 
                                              history_action_hash_set , 
                                              present_state_hash_set , 
                                              future_action_hash_set , 
                                              future_reward_hash_set , 
                                              future_state_hash_set ,
                                              history_state_list,
                                              history_action_list,
                                              present_state_list,
                                              future_action_list,
                                              future_reward_list,
                                              future_state_list):

    new_history_state_list, new_history_action_list, new_present_state_list, new_future_action_list, new_future_reward_list, new_future_state_list = [], [], [], [], [], []

    for i in range(len(present_state_list)):
        history_state  = history_state_list [i]
        history_action = history_action_list[i]
        present_state  = present_state_list [i]
        future_action  = future_action_list [i]
        future_reward  = future_reward_list [i]
        future_state   = future_state_list  [i]

        hs_hash = fast_hash_tensor(history_state )
        ha_hash = fast_hash_tensor(history_action)
        ps_hash = fast_hash_tensor(present_state )
        fa_hash = fast_hash_tensor(future_action )
        fr_hash = fast_hash_tensor(future_reward )
        fs_hash = fast_hash_tensor(future_state  )

        if (hs_hash not in history_state_hash_set  or
            ha_hash not in history_action_hash_set or
            ps_hash not in present_state_hash_set  or
            fa_hash not in future_action_hash_set  or
            fr_hash not in future_reward_hash_set  or
            fs_hash not in future_state_hash_set):

            new_history_state_list .append(history_state .unsqueeze(0))
            new_history_action_list.append(history_action.unsqueeze(0))
            new_present_state_list .append(present_state .unsqueeze(0))
            new_future_action_list .append(future_action .unsqueeze(0))
            new_future_reward_list .append(future_reward .unsqueeze(0))
            new_future_state_list  .append(future_state  .unsqueeze(0))

            history_state_hash_set .add(hs_hash)
            history_action_hash_set.add(ha_hash)
            present_state_hash_set .add(ps_hash)
            future_action_hash_set .add(fa_hash)
            future_reward_hash_set .add(fr_hash)
            future_state_hash_set  .add(fs_hash)

    if new_present_state_list:
        history_state_stack  = torch.cat([history_state_stack ] + new_history_state_list , dim=0)
        history_action_stack = torch.cat([history_action_stack] + new_history_action_list, dim=0)
        present_state_stack  = torch.cat([present_state_stack ] + new_present_state_list , dim=0)
        future_action_stack  = torch.cat([future_action_stack ] + new_future_action_list , dim=0)
        future_reward_stack  = torch.cat([future_reward_stack ] + new_future_reward_list , dim=0)
        future_state_stack   = torch.cat([future_state_stack  ] + new_future_state_list  , dim=0)

    return history_state_stack, history_action_stack, present_state_stack, future_action_stack, future_reward_stack, future_state_stack,\
           history_state_hash_set, history_action_hash_set, present_state_hash_set, future_action_hash_set, future_reward_hash_set, future_state_hash_set 




def obtain_priority_probability_(model, dataset, batch_size, device, param=1.0, min_prob=0.01):

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    reward_list = []
    for _, _, _, _, future_reward, _ in data_loader:
        reward_list.append(future_reward[:, -1:, :].detach()) # future_reward shape: [batch, 1, reward_dim]
    rewards = torch.cat(reward_list, dim=0).to(device)  # shape [N, 1, reward_dim] or [N, 1]
    rewards = (rewards + 1) / 2 # normalize [-1, 1] to [0, 1]
    rewards = rewards.mean(dim=tuple(range(1, rewards.dim())))

    # 🔹 Type 1: weight by reward
    # sample_weights = rewards ** param
    # sample_weights = torch.clamp(sample_weights, min=min_prob)
    # probabilities  = sample_weights / sample_weights.sum()

    # 🔹 Type 2: inverse weight by count
    unique_rewards, counts = torch.unique(rewards, return_counts=True)
    inv_freq = 1.0 / counts.float()
    inv_freq = inv_freq ** param  # power sparse reward
    reward_to_invfreq = dict(zip(unique_rewards.tolist(), inv_freq.tolist()))
    sample_weights = torch.tensor([reward_to_invfreq[r.item()] for r in rewards], device=device)
    probabilities = torch.clamp(sample_weights, min=min_prob)
    probabilities = probabilities / probabilities.sum()

    # 🔹 Type 3: weight by category
    # unique_rewards, counts = torch.unique(rewards, return_counts=True)
    # num_unique = len(unique_rewards)
    # class_total_prob = 1.0 / num_unique
    # reward_to_prob = {u.item(): class_total_prob / c.item()
    #                   for u, c in zip(unique_rewards, counts)}
    # sample_probs = torch.tensor([reward_to_prob[r.item()] for r in rewards], device=device)
    # probabilities = torch.clamp(sample_probs, min=min_prob)
    # probabilities = probabilities / probabilities.sum()

    # 🔹 Type 4: normalize then softmax
    # rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
    # logits = rewards / param
    # sample_weights = torch.softmax(logits, dim=0)
    # sample_weights = torch.clamp(sample_weights, min=min_prob)
    # probabilities  = sample_weights / sample_weights.sum()

    return probabilities 

def obtain_priority_probability__(model, dataset, batch_size, device, param=1.0, min_prob=0.01):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    reward_list = []
    for history_state, history_action, present_state, _, future_reward, _ in data_loader:
        history_state  = history_state .reshape(history_state.size(0), -1)
        history_action = history_action.reshape(history_action.size(0), -1)
        present_state  = present_state.reshape(present_state.size(0), -1)
        future_reward  = future_reward[:, -1:, :].reshape(future_reward.size(0), -1)  
        combined = torch.cat((history_state, history_action, present_state, future_reward), dim=1)
        reward_list.append(combined.detach())

    rewards = torch.cat(reward_list, dim=0).to(device)  # [N, D]

    # 🔹 unique rewards by row
    unique_rewards, inverse_indices, counts = torch.unique(rewards, dim=0, return_inverse=True, return_counts=True)

    # 🔹 inverse frequency weighting
    inv_freq = (1.0 / counts.float()) ** param
    sample_weights = inv_freq[inverse_indices]  # map back to each sample
    probabilities = torch.clamp(sample_weights, min=min_prob)
    probabilities = probabilities / probabilities.sum()

    return probabilities

def obtain_priority_probability(model, dataset, batch_size, device, param=1.0, min_prob=0.01):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    # 🔹
    history_state_list, history_action_list, present_state_list, future_reward_list = [], [], [], []
    for history_state, history_action, present_state, _, future_reward, _ in data_loader:
        history_state  = history_state .reshape(history_state.size(0), -1)
        history_action = history_action.reshape(history_action.size(0), -1)
        present_state  = present_state.reshape(present_state.size(0), -1)
        future_reward  = future_reward [:, -1:, :].reshape(future_reward.size(0), -1)

        history_state_list.append(history_state.detach())
        history_action_list.append(history_action.detach())
        present_state_list.append(present_state.detach())
        future_reward_list.append(future_reward.detach())

    # 🔹🔹
    history_state  = torch.cat(history_state_list, dim=0).to(device)
    history_action = torch.cat(history_action_list, dim=0).to(device)
    present_state  = torch.cat(present_state_list, dim=0).to(device)
    future_reward  = torch.cat(future_reward_list, dim=0).to(device)

    # 🔹🔹🔹
    state_action = torch.cat((history_state, history_action, present_state), dim=1)  # [N, D]
    sa_unique, sa_inverse_idx, sa_counts = torch.unique(
        state_action, dim=0, return_inverse=True, return_counts=True
    )
    num_groups = sa_unique.size(0)
    group_base_prob = torch.ones(num_groups, device=device) / num_groups  

    # 🔹🔹🔹🔹
    sample_weights = torch.zeros_like(sa_inverse_idx, dtype=torch.float, device=device)
    for g in range(num_groups):
        mask = (sa_inverse_idx == g)
        rewards_g = future_reward[mask]  # shape: [N_g]
        rewards_g = (rewards_g + 1) / 2 # normalize [-1, 1] to [0, 1]
        rewards_g = rewards_g.mean(dim=tuple(range(1, rewards_g.dim())))
        rewards_g = rewards_g ** param
        rewards_g = torch.clamp(rewards_g, min=min_prob)
        local_base_prob  = rewards_g / rewards_g.sum()
        sample_weights[mask] = group_base_prob[g] * local_base_prob
    
    # 🔹🔹🔹🔹🔹 
    probabilities = torch.clamp(sample_weights, min=min_prob)
    probabilities = probabilities / probabilities.sum()

    return probabilities

def update_model_per(itrtn_for_learning,
                     dataset,
                     model,
                     batch_size,
                     priority_probability):
    
    device = next(model.parameters()).device

    for _ in range(itrtn_for_learning):

        final_indices  = torch.multinomial(priority_probability, batch_size, replacement=True)
        final_indices  = final_indices.cpu().tolist()

        batch_samples  = [dataset[i] for i in final_indices]
        history_state, history_action, present_state, future_action, future_reward, future_state = zip(*batch_samples)
        history_state  = torch.stack(history_state ).to(device)
        history_action = torch.stack(history_action).to(device)
        present_state  = torch.stack(present_state ).to(device)
        future_action  = torch.stack(future_action ).to(device)
        future_reward  = torch.stack(future_reward ).to(device)
        future_state   = torch.stack(future_state  ).to(device)

        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        envisaged_reward, \
        envisaged_state             = model.forward_(history_state, history_action, present_state, future_state, future_action)
        total_loss                  = loss_function(envisaged_reward, future_reward) + loss_function(envisaged_state, future_state )
        total_loss.backward()     
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip_value)
        selected_optimizer.step() 

    return model




def update_model(itrtn_for_learning,
                 dataset,
                 model,
                 batch_size,
                 param):
    
    device = next(model.parameters()).device

    for _ in range(itrtn_for_learning):

        final_indices  = random.choices(range(len(dataset)), k=batch_size)

        batch_samples  = [dataset[i] for i in final_indices]
        history_state, history_action, present_state, future_action, future_reward, future_state = zip(*batch_samples)
        history_state  = torch.stack(history_state ).to(device)
        history_action = torch.stack(history_action).to(device)
        present_state  = torch.stack(present_state ).to(device)
        future_action  = torch.stack(future_action ).to(device)
        future_reward  = torch.stack(future_reward ).to(device)
        future_state   = torch.stack(future_state  ).to(device)
 
        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        envisaged_reward, \
        envisaged_state             = model.forward_(history_state, history_action, present_state, future_state, future_action)
        total_loss                  = loss_function(envisaged_reward, future_reward) + loss_function(envisaged_state, future_state )
        total_loss.backward()   

        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip_value)
        selected_optimizer.step() 

    return model




def update_model_list(itrtn_for_learning,
                      dataset,
                      model_list,
                      batch_size,
                      param,
                      PER):
    if not PER:
        for i, model in enumerate(tqdm(model_list, desc="Updating models")):
            model_list[i] = update_model(itrtn_for_learning,
                                         dataset,
                                         model,
                                         batch_size,
                                         param)
    else:
        model  = model_list[0]
        device = next(model.parameters()).device
        priority_probability = obtain_priority_probability(model, dataset, len(dataset), device, param=param)
        for i, model in enumerate(tqdm(model_list, desc="Updating models")):
            model_list[i] = update_model_per(itrtn_for_learning,
                                             dataset,
                                             model,
                                             batch_size,
                                             priority_probability)
    return model_list




def limit_buffer(history_state_stack, 
                 history_action_stack,
                 present_state_stack, 
                 future_action_stack,
                 future_reward_stack, 
                 future_state_stack,
                 history_state_hash_set , 
                 history_action_hash_set , 
                 present_state_hash_set , 
                 future_action_hash_set , 
                 future_reward_hash_set , 
                 future_state_hash_set ,
                 buffer_limit ):

    n = len(present_state_stack)
    probability = torch.ones(n) / n
    indices_to_keep = torch.multinomial(probability, min(buffer_limit, n), replacement=False).tolist()

    # slice tensor buffers
    history_state_stack  = history_state_stack [indices_to_keep]
    history_action_stack = history_action_stack[indices_to_keep]
    present_state_stack  = present_state_stack [indices_to_keep]
    future_action_stack  = future_action_stack [indices_to_keep]
    future_reward_stack  = future_reward_stack [indices_to_keep]
    future_state_stack   = future_state_stack  [indices_to_keep]

    hs_hash_set = set()
    ha_hash_set = set()
    ps_hash_set = set()
    fa_hash_set = set()
    fr_hash_set = set()
    fs_hash_set = set()
    for i in range(len(present_state_stack)):
        hs_hash_set.add(fast_hash_tensor(history_state_stack [i]))
        ha_hash_set.add(fast_hash_tensor(history_action_stack[i]))
        ps_hash_set.add(fast_hash_tensor(present_state_stack [i]))
        fa_hash_set.add(fast_hash_tensor(future_action_stack [i]))
        fr_hash_set.add(fast_hash_tensor(future_reward_stack [i]))
        fs_hash_set.add(fast_hash_tensor(future_state_stack  [i]))

    history_state_hash_set  = history_state_hash_set  & hs_hash_set
    history_action_hash_set = history_action_hash_set & ha_hash_set
    present_state_hash_set  = present_state_hash_set  & ps_hash_set
    future_action_hash_set  = future_action_hash_set  & fa_hash_set
    future_reward_hash_set  = future_reward_hash_set  & fr_hash_set
    future_state_hash_set   = future_state_hash_set   & fs_hash_set
    
    return history_state_stack, history_action_stack, present_state_stack, future_action_stack, future_reward_stack, future_state_stack,\
           history_state_hash_set, history_action_hash_set, present_state_hash_set, future_action_hash_set, future_reward_hash_set, future_state_hash_set 




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)




def save_buffer_to_pickle(filename, *list):
    with open(filename, 'wb') as file:
        dill.dump(list, file)


