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
    return torch.load(filename)




def retrieve_history_and_present(state_list, action_list, reward_list, history_size, frame_skip, device):
    if history_size != 0:
        history_size     *= frame_skip
        history_reward    = torch.stack(reward_list[-history_size-1:-1: frame_skip], dim=0).unsqueeze(0).to(device, non_blocking=True)
        history_state     = torch.stack(state_list [-history_size-1:-1: frame_skip], dim=0).unsqueeze(0).to(device, non_blocking=True)
        history_action    = torch.stack(action_list[-history_size  :  : frame_skip], dim=0).unsqueeze(0).to(device, non_blocking=True)
    else:
        history_reward    = torch.empty(0, 0, 0).to(device, non_blocking=True)
        history_state     = torch.empty(0, 0, 0).to(device, non_blocking=True)
        history_action    = torch.empty(0, 0, 0).to(device, non_blocking=True)
    present_reward    = reward_list[-1].unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
    present_state     = state_list [-1].unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
    return history_reward, history_state, history_action, present_reward, present_state




def initialize_action(shape, device, mean=0.0, std=0.0):
    return torch.normal(
        mean=mean,
        std=std,
        size=shape,
        device=device
    )




def initialize_desired_reward(shape, device):
    return torch.ones(shape).to(device, non_blocking=True)




def update_future_action(itrtn_for_planning,
                         model_list,
                         history_reward,
                         history_state,
                         history_action,
                         present_reward,
                         present_state,
                         present_action,
                         future_action,
                         desired_reward,
                         beta):

    device  = next(model_list[0].parameters()).device
    device_ = history_reward.device

    history_reward = history_reward.to(device)
    history_state  = history_state .to(device)
    history_action = history_action.to(device)
    present_reward = present_reward.to(device)
    present_state  = present_state .to(device)
    present_action = present_action.to(device)
    future_action  = future_action .to(device)
    desired_reward = desired_reward.to(device)

    present_action     = torch.nn.Parameter(present_action)
    future_action      = torch.nn.Parameter(future_action)
    selected_optimizer = torch.optim.SGD([present_action, future_action], lr=beta)
    # selected_optimizer = torch.optim.Adam([present_action, future_action], lr=beta)

    for _ in range(itrtn_for_planning):

        model              = random.choice(model_list)

        model.train()
        for p in model.parameters():
            p.requires_grad_(False)
        selected_optimizer.zero_grad()

        present_a = torch.tanh(present_action)
        future_a  = torch.tanh(future_action)

        loss_function      = model.loss_function
        envisaged_reward, \
        envisaged_state    = model.forward(history_reward, history_state, history_action, present_reward, present_state, present_a, None, None, future_a)
        total_loss         = loss_function(envisaged_reward, desired_reward)
        total_loss.backward()

        selected_optimizer.step()

    present_action = present_action.detach().to(device_, non_blocking=True)
    future_action  = future_action.detach().to(device_, non_blocking=True)

    return present_action, future_action




def sequentialize(state_list, action_list, reward_list, history_size, future_size, skip):

    device              = state_list[0].device
    torch_empty         = torch.empty(0, 0, 0).to(device, non_blocking=True)

    history_reward_list = []
    history_state_list  = []
    history_action_list = []
    present_reward_list = []
    present_state_list  = []
    present_action_list = []
    future_reward_list  = []
    future_state_list   = []
    future_action_list  = []

    if history_size > 0:

        history_size *= skip
        future_size  *= skip

        for i in range(max(0, len(reward_list) - history_size - future_size - skip + 1)):

            node  = i + history_size

            history_reward_list.append(      torch.stack(reward_list[ i : node                                            :  skip  ], dim=0)          )
            history_state_list.append (      torch.stack(state_list [ i : node                                            :  skip  ], dim=0)          )
            history_action_list.append(      torch.stack(action_list[ i : node                                            :  skip  ], dim=0)          )
            present_reward_list.append(                  reward_list[     node                                                     ].unsqueeze(0)      )
            present_state_list.append (                  state_list [     node                                                     ].unsqueeze(0)      )
            present_action_list.append(                  action_list[     node                                                     ].unsqueeze(0)      )
            future_reward_list.append (      torch.stack(reward_list[     node            : node + future_size     + skip :  skip  ], dim=0)          )
            future_state_list.append  (      torch.stack(state_list [     node     + skip : node + future_size + 2 * skip :  skip  ], dim=0)          )
            future_action_list.append (      torch.stack(action_list[     node     + skip : node + future_size     + skip :  skip  ], dim=0)          )

    else:

        for i in range(max(0, len(reward_list) - future_size - skip + 1)):

            node = i

            history_reward_list.append(                  torch_empty                                                                                   )
            history_state_list.append (                  torch_empty                                                                                   )
            history_action_list.append(                  torch_empty                                                                                   )
            present_reward_list.append(                 (reward_list[     node - 1                                                 ] if node > 0 else torch.zeros_like(reward_list[0]).fill_(-1)).unsqueeze(0) )
            present_state_list.append (                  state_list [     node                                                     ].unsqueeze(0)      )
            present_action_list.append(                  action_list[     node                                                     ].unsqueeze(0)      )
            future_reward_list.append (      torch.stack(reward_list[     node            : node + future_size     + skip :  skip  ], dim=0)          )
            future_state_list.append  (      torch.stack(state_list [     node     + skip : node + future_size + 2 * skip :  skip  ], dim=0)          )
            future_action_list.append (      torch.stack(action_list[     node     + skip : node + future_size     + skip :  skip  ], dim=0)          )

    return history_reward_list, history_state_list, history_action_list, \
           present_reward_list, present_state_list, present_action_list, \
           future_reward_list, future_state_list, future_action_list




def fast_hash_tensor(tensor):
    arr = tensor.detach().cpu().view(-1)
    sample = arr.numpy().tobytes()
    return zlib.adler32(sample)

def update_long_term_experience_replay_buffer(history_reward_stack,
                                              history_state_stack,
                                              history_action_stack,
                                              present_reward_stack,
                                              present_state_stack,
                                              present_action_stack,
                                              future_reward_stack,
                                              future_state_stack,
                                              future_action_stack,
                                              history_reward_hash_set,
                                              history_state_hash_set,
                                              history_action_hash_set,
                                              present_reward_hash_set,
                                              present_state_hash_set,
                                              present_action_hash_set,
                                              future_reward_hash_set,
                                              future_state_hash_set,
                                              future_action_hash_set,
                                              history_reward_list,
                                              history_state_list,
                                              history_action_list,
                                              present_reward_list,
                                              present_state_list,
                                              present_action_list,
                                              future_reward_list,
                                              future_state_list,
                                              future_action_list):

    new_history_reward_list, new_history_state_list, new_history_action_list = [], [], []
    new_present_reward_list, new_present_state_list, new_present_action_list = [], [], []
    new_future_reward_list,  new_future_state_list,  new_future_action_list  = [], [], []

    for i in range(len(present_state_list)):
        history_reward = history_reward_list[i]
        history_state  = history_state_list [i]
        history_action = history_action_list[i]
        present_reward = present_reward_list[i]
        present_state  = present_state_list [i]
        present_action = present_action_list[i]
        future_reward  = future_reward_list [i]
        future_state   = future_state_list  [i]
        future_action  = future_action_list [i]

        hr_hash = fast_hash_tensor(history_reward)
        hs_hash = fast_hash_tensor(history_state )
        ha_hash = fast_hash_tensor(history_action)
        pr_hash = fast_hash_tensor(present_reward)
        ps_hash = fast_hash_tensor(present_state )
        pa_hash = fast_hash_tensor(present_action)
        fr_hash = fast_hash_tensor(future_reward )
        fs_hash = fast_hash_tensor(future_state  )
        fa_hash = fast_hash_tensor(future_action )

        if (hr_hash not in history_reward_hash_set or
            hs_hash not in history_state_hash_set  or
            ha_hash not in history_action_hash_set or
            pr_hash not in present_reward_hash_set or
            ps_hash not in present_state_hash_set  or
            pa_hash not in present_action_hash_set or
            fr_hash not in future_reward_hash_set  or
            fs_hash not in future_state_hash_set   or
            fa_hash not in future_action_hash_set):

            new_history_reward_list.append(history_reward.unsqueeze(0))
            new_history_state_list .append(history_state .unsqueeze(0))
            new_history_action_list.append(history_action.unsqueeze(0))
            new_present_reward_list.append(present_reward.unsqueeze(0))
            new_present_state_list .append(present_state .unsqueeze(0))
            new_present_action_list.append(present_action.unsqueeze(0))
            new_future_reward_list .append(future_reward .unsqueeze(0))
            new_future_state_list  .append(future_state  .unsqueeze(0))
            new_future_action_list .append(future_action .unsqueeze(0))

            history_reward_hash_set.add(hr_hash)
            history_state_hash_set .add(hs_hash)
            history_action_hash_set.add(ha_hash)
            present_reward_hash_set.add(pr_hash)
            present_state_hash_set .add(ps_hash)
            present_action_hash_set.add(pa_hash)
            future_reward_hash_set .add(fr_hash)
            future_state_hash_set  .add(fs_hash)
            future_action_hash_set .add(fa_hash)

    if new_present_state_list:
        history_reward_stack = torch.cat([history_reward_stack] + new_history_reward_list, dim=0)
        history_state_stack  = torch.cat([history_state_stack ] + new_history_state_list , dim=0)
        history_action_stack = torch.cat([history_action_stack] + new_history_action_list, dim=0)
        present_reward_stack = torch.cat([present_reward_stack] + new_present_reward_list, dim=0)
        present_state_stack  = torch.cat([present_state_stack ] + new_present_state_list , dim=0)
        present_action_stack = torch.cat([present_action_stack] + new_present_action_list, dim=0)
        future_reward_stack  = torch.cat([future_reward_stack ] + new_future_reward_list , dim=0)
        future_state_stack   = torch.cat([future_state_stack  ] + new_future_state_list  , dim=0)
        future_action_stack  = torch.cat([future_action_stack ] + new_future_action_list , dim=0)

    return history_reward_stack, history_state_stack, history_action_stack, \
           present_reward_stack, present_state_stack, present_action_stack, \
           future_reward_stack, future_state_stack, future_action_stack, \
           history_reward_hash_set, history_state_hash_set, history_action_hash_set, \
           present_reward_hash_set, present_state_hash_set, present_action_hash_set, \
           future_reward_hash_set, future_state_hash_set, future_action_hash_set




def obtain_priority_probability(model, dataset, batch_size, PER_epsilon, PER_exponent, device):

    data_loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    TD_error_list = []

    for history_reward, history_state, history_action, present_reward, present_state, present_action, future_reward, future_state, future_action in data_loader:

        history_reward = history_reward.to(device)
        history_state  = history_state .to(device)
        history_action = history_action.to(device)
        present_reward = present_reward.to(device)
        present_state  = present_state .to(device)
        present_action = present_action.to(device)
        future_reward  = future_reward .to(device)
        future_state   = future_state  .to(device)
        future_action  = future_action .to(device)

        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function                 = model.loss_function_
        envisaged_reward, \
        envisaged_state               = model.forward_(history_reward, history_state, history_action, present_reward, present_state, present_action, future_reward, future_state, future_action)
        total_loss                    = torch.sum(torch.abs(loss_function(envisaged_reward[:, :, :], future_reward[:, :, :]) ), dim=(1, 2)) + \
                                        torch.sum(torch.abs(loss_function(envisaged_state [:, :, :], future_state [:, :, :]) ), dim=(1, 2))
        TD_error_list.append(total_loss.detach())

    obsolute_TD_error    = torch.cat(TD_error_list, dim=0).to(device)
    priority             = obsolute_TD_error + PER_epsilon
    exponent_priority    = priority ** PER_exponent
    probabilities        = exponent_priority / torch.sum(exponent_priority)

    return probabilities

def update_model_per(itrtn_for_learning,
                     dataset,
                     model,
                     batch_size):

    device         = next(model.parameters()).device

    for _ in range(itrtn_for_learning):

        priority_batch_size  = 50
        priority_epsilon     = 1e-10
        priority_exponent    = 1
        priority_probability = obtain_priority_probability(model, dataset, priority_batch_size, priority_epsilon, priority_exponent, device)
        final_indices        = torch.multinomial(priority_probability, batch_size, replacement=False)

        batch_samples  = [dataset[i] for i in final_indices]
        history_reward, history_state, history_action, present_reward, present_state, present_action, future_reward, future_state, future_action = zip(*batch_samples)
        history_reward = torch.stack(history_reward).to(device)
        history_state  = torch.stack(history_state ).to(device)
        history_action = torch.stack(history_action).to(device)
        present_reward = torch.stack(present_reward).to(device)
        present_state  = torch.stack(present_state ).to(device)
        present_action = torch.stack(present_action).to(device)
        future_reward  = torch.stack(future_reward ).to(device)
        future_state   = torch.stack(future_state  ).to(device)
        future_action  = torch.stack(future_action ).to(device)

        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        envisaged_reward, \
        envisaged_state             = model.forward_(history_reward, history_state, history_action, present_reward, present_state, present_action, future_reward, future_state, future_action)
        total_loss                  = loss_function(envisaged_reward, future_reward) + loss_function(envisaged_state, future_state)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip_value)
        selected_optimizer.step()

    return model

# def obtain_priority_probability(model, dataset, device):
# 
#     data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, pin_memory=True, num_workers=0)
#     
#     reward_list = []
# 
#     for history_state, history_action, present_state, _, future_reward, _ in data_loader:
#         # history_state  = history_state.reshape(history_state.size(0), -1)
#         # history_action = history_action.reshape(history_action.size(0), -1)
#         # present_state  = present_state.reshape(present_state.size(0), -1)
#         future_reward  = future_reward[:, -1:, :].reshape(future_reward.size(0), -1)  
#         # combined = torch.cat((history_state, history_action, present_state, future_reward), dim=1)
#         combined = future_reward
#         reward_list.append(combined.detach())
# 
#     rewards = torch.cat(reward_list, dim=0).to(device)  # [N, D]
# 
#     # 🔹 unique rewards by row
#     unique_rewards, inverse_indices, counts = torch.unique(rewards, dim=0, return_inverse=True, return_counts=True)
# 
#     num_classes   = unique_rewards.size(0)
#     probabilities = 1.0 / (num_classes * counts[inverse_indices].float())
#     probabilities = probabilities / probabilities.sum()
# 
#     return probabilities
# 
# def update_model_per(itrtn_for_learning,
#                      dataset,
#                      model,
#                      batch_size):
#         
#     device         = next(model.parameters()).device
#     priority_probability = obtain_priority_probability(model, dataset, device)
# 
#     for _ in range(itrtn_for_learning):
# 
#         final_indices        = torch.multinomial(priority_probability, batch_size, replacement=False)
# 
#         batch_samples  = [dataset[i] for i in final_indices]
#         history_state, history_action, present_state, future_action, future_reward, future_state = zip(*batch_samples)
#         history_state  = torch.stack(history_state ).to(device)
#         history_action = torch.stack(history_action).to(device)
#         present_state  = torch.stack(present_state ).to(device)
#         future_action  = torch.stack(future_action ).to(device)
#         future_reward  = torch.stack(future_reward ).to(device)
#         future_state   = torch.stack(future_state  ).to(device)
# 
#         model.train()
#         selected_optimizer = model.selected_optimizer
#         selected_optimizer.zero_grad()
# 
#         loss_function               = model.loss_function
#         envisaged_reward, \
#         envisaged_state             = model.forward_(history_state, history_action, present_state, future_state, future_action)
#         total_loss                  = loss_function(envisaged_reward, future_reward) + loss_function(envisaged_state, future_state )
#         total_loss.backward()     
#         
#         torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip_value)
#         selected_optimizer.step() 
# 
#     return model




def update_model(itrtn_for_learning,
                 dataset,
                 model,
                 batch_size):

    device = next(model.parameters()).device

    for _ in range(itrtn_for_learning):

        final_indices  = random.choices(range(len(dataset)), k=batch_size)

        batch_samples  = [dataset[i] for i in final_indices]
        history_reward, history_state, history_action, present_reward, present_state, present_action, future_reward, future_state, future_action = zip(*batch_samples)

        history_reward = torch.stack(history_reward).to(device)
        history_state  = torch.stack(history_state ).to(device)
        history_action = torch.stack(history_action).to(device)
        present_reward = torch.stack(present_reward).to(device)
        present_state  = torch.stack(present_state ).to(device)
        present_action = torch.stack(present_action).to(device)
        future_reward  = torch.stack(future_reward ).to(device)
        future_state   = torch.stack(future_state  ).to(device)
        future_action  = torch.stack(future_action ).to(device)

        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        envisaged_reward, \
        envisaged_state             = model.forward_(history_reward, history_state, history_action, present_reward, present_state, present_action, future_reward, future_state, future_action)
        total_loss                  = loss_function(envisaged_reward, future_reward) + loss_function(envisaged_state, future_state)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip_value)
        selected_optimizer.step()

    return model




def update_model_list(itrtn_for_learning,
                      dataset,
                      model_list,
                      PER,
                      batch_size = 1):
    if not PER:
        for i, model in enumerate(tqdm(model_list, desc="Updating models")):
            model_list[i] = update_model(itrtn_for_learning,
                                         dataset,
                                         model,
                                         batch_size)
    else:
        for i, model in enumerate(tqdm(model_list, desc="Updating models")):
            model_list[i] = update_model_per(itrtn_for_learning,
                                             dataset,
                                             model,
                                             batch_size)
    return model_list




def limit_buffer(history_reward_stack,
                 history_state_stack,
                 history_action_stack,
                 present_reward_stack,
                 present_state_stack,
                 present_action_stack,
                 future_reward_stack,
                 future_state_stack,
                 future_action_stack,
                 history_reward_hash_set,
                 history_state_hash_set,
                 history_action_hash_set,
                 present_reward_hash_set,
                 present_state_hash_set,
                 present_action_hash_set,
                 future_reward_hash_set,
                 future_state_hash_set,
                 future_action_hash_set,
                 buffer_limit):

    n = len(present_state_stack)
    probability = torch.ones(n) / n
    indices_to_keep = torch.multinomial(probability, min(buffer_limit, n), replacement=False).tolist()

    # slice tensor buffers
    history_reward_stack = history_reward_stack[indices_to_keep]
    history_state_stack  = history_state_stack [indices_to_keep]
    history_action_stack = history_action_stack[indices_to_keep]
    present_reward_stack = present_reward_stack[indices_to_keep]
    present_state_stack  = present_state_stack [indices_to_keep]
    present_action_stack = present_action_stack[indices_to_keep]
    future_reward_stack  = future_reward_stack [indices_to_keep]
    future_state_stack   = future_state_stack  [indices_to_keep]
    future_action_stack  = future_action_stack [indices_to_keep]

    hr_hash_set = set()
    hs_hash_set = set()
    ha_hash_set = set()
    pr_hash_set = set()
    ps_hash_set = set()
    pa_hash_set = set()
    fr_hash_set = set()
    fs_hash_set = set()
    fa_hash_set = set()
    for i in range(len(present_state_stack)):
        hr_hash_set.add(fast_hash_tensor(history_reward_stack[i]))
        hs_hash_set.add(fast_hash_tensor(history_state_stack [i]))
        ha_hash_set.add(fast_hash_tensor(history_action_stack[i]))
        pr_hash_set.add(fast_hash_tensor(present_reward_stack[i]))
        ps_hash_set.add(fast_hash_tensor(present_state_stack [i]))
        pa_hash_set.add(fast_hash_tensor(present_action_stack[i]))
        fr_hash_set.add(fast_hash_tensor(future_reward_stack [i]))
        fs_hash_set.add(fast_hash_tensor(future_state_stack  [i]))
        fa_hash_set.add(fast_hash_tensor(future_action_stack [i]))

    history_reward_hash_set = history_reward_hash_set & hr_hash_set
    history_state_hash_set  = history_state_hash_set  & hs_hash_set
    history_action_hash_set = history_action_hash_set & ha_hash_set
    present_reward_hash_set = present_reward_hash_set & pr_hash_set
    present_state_hash_set  = present_state_hash_set  & ps_hash_set
    present_action_hash_set = present_action_hash_set & pa_hash_set
    future_reward_hash_set  = future_reward_hash_set  & fr_hash_set
    future_state_hash_set   = future_state_hash_set   & fs_hash_set
    future_action_hash_set  = future_action_hash_set  & fa_hash_set

    return history_reward_stack, history_state_stack, history_action_stack, \
           present_reward_stack, present_state_stack, present_action_stack, \
           future_reward_stack, future_state_stack, future_action_stack, \
           history_reward_hash_set, history_state_hash_set, history_action_hash_set, \
           present_reward_hash_set, present_state_hash_set, present_action_hash_set, \
           future_reward_hash_set, future_state_hash_set, future_action_hash_set




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)




def save_buffer_to_pickle(filename, *list):
    torch.save(list, filename)











# def get_latent(model, dataset, device):
# 
#     data_loader  = DataLoader(dataset, batch_size = len(dataset), shuffle=False, pin_memory=True, num_workers=0)
#     
#     TD_error_list = []
# 
#     for history_state, history_action, present_state, future_action, future_reward, future_state in data_loader:
# 
#         history_state  = history_state .to(device)
#         history_action = history_action.to(device)
#         present_state  = present_state .to(device)
#         future_action  = future_action .to(device)
#         future_reward  = future_reward .to(device)
#         future_state   = future_state  .to(device)
# 
#         model.train()
#         selected_optimizer = model.selected_optimizer
#         selected_optimizer.zero_grad()
# 
#         loss_function                 = model.loss_function_
#         latent                        = model.latent(history_state, history_action, present_state, future_state, future_action)
# 
#     return latent
# 
# 
# 
# 
# def obtain_obsolute_TD_error(model, dataset, latent, device):
# 
#     data_loader  = DataLoader(dataset, batch_size = len(dataset), shuffle=False, pin_memory=True, num_workers=0)
#     
#     TD_error_list = []
# 
#     for history_state, history_action, present_state, future_action, future_reward, future_state in data_loader:
# 
#         history_state  = history_state .to(device)
#         history_action = history_action.to(device)
#         present_state  = present_state .to(device)
#         future_action  = future_action .to(device)
#         future_reward  = future_reward .to(device)
#         future_state   = future_state  .to(device)
# 
#         model.train()
#         selected_optimizer = model.selected_optimizer
#         selected_optimizer.zero_grad()
# 
#         loss_function                 = model.loss_function_
#         envisaged_reward, \
#         envisaged_state               = model.latentforward(latent, future_action)
#         total_loss                    = torch.sum(torch.abs(loss_function(envisaged_reward[:, :, :], future_reward[:, :, :]) ), dim=(1, 2)) + \
#                                         torch.sum(torch.abs(loss_function(envisaged_state [:, :, :], future_state [:, :, :]) ), dim=(1, 2))
#         TD_error_list.append(total_loss.detach())  
# 
#     TD_error = torch.cat(TD_error_list, dim=0).to(device)
# 
#     return TD_error
# 
# 
# 
# 
# def update_model_per(itrtn_for_learning,
#                      dataset,
#                      model,
#                      batch_size,
#                      param):
#         
#     device         = next(model.parameters()).device
#     PER_epsilon    = 1e-10
#     PER_exponent   = param
#     latent         = get_latent(model, dataset, device)
# 
#     for _ in range(itrtn_for_learning):
# 
#         obsolute_TD_error    = obtain_obsolute_TD_error(model, dataset, latent, device)
#         priority             = obsolute_TD_error + PER_epsilon
#         exponent_priority    = priority ** PER_exponent
#         priority_probability = exponent_priority / torch.sum(exponent_priority)
#         final_indices        = torch.multinomial(priority_probability, batch_size, replacement=False)
# 
#         batch_samples  = [dataset[i] for i in final_indices]
#         history_state, history_action, present_state, future_action, future_reward, future_state = zip(*batch_samples)
#         history_state  = torch.stack(history_state ).to(device)
#         history_action = torch.stack(history_action).to(device)
#         present_state  = torch.stack(present_state ).to(device)
#         future_action  = torch.stack(future_action ).to(device)
#         future_reward  = torch.stack(future_reward ).to(device)
#         future_state   = torch.stack(future_state  ).to(device)
# 
#         model.train()
#         selected_optimizer = model.selected_optimizer
#         selected_optimizer.zero_grad()
# 
#         loss_function               = model.loss_function
#         envisaged_reward, \
#         envisaged_state             = model.forward_(history_state, history_action, present_state, future_state, future_action)
#         total_loss                  = loss_function(envisaged_reward, future_reward) + loss_function(envisaged_state, future_state )
#         total_loss.backward()     
#         
#         torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip_value)
#         selected_optimizer.step() 
# 
#     return model
# 
# 
# 
# 