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

from itertools import cycle, islice
from collections import deque




class make_buffer:
    def __init__(self, max_episode=1000, device="cpu", alpha=10.0, recent_n=50):
        self.buffer = deque(maxlen=max_episode)
        self.device = device
        self.alpha  = alpha 
        self.recent_n = recent_n

    def to(self, device):
        """
        Move all tensors in the buffer to a new device (e.g., 'cuda:0').
        """
        # device = torch.device(device)
        for episode in self.buffer:
            for transition in episode["transitions"]:
                for key, value in transition.items():
                    # if torch.is_tensor(value):
                    transition[key] = value.to(device)
        self.device = device
        return self
    
    def add_episode(self, episode_counter, history_state_list, history_action_list, present_state_list, future_action_list, future_reward_list, future_state_list):
        transitions = []
        for i in range(len(present_state_list)):
            transition = {
                "history_state" : history_state_list [i].to(self.device),
                "history_action": history_action_list[i].to(self.device),
                "present_state" : present_state_list [i].to(self.device),
                "future_action" : future_action_list [i].to(self.device),
                "future_reward" : future_reward_list [i].to(self.device),
                "future_state"  : future_state_list  [i].to(self.device)
            }
            transitions.append(transition)
        self.buffer.append({
            "episode_counter": episode_counter,
            "transitions": transitions
        })

    def _get_time_weights(self):
        N = len(self.buffer)
        indices = torch.arange(N, dtype=torch.float32)
        weights = torch.exp(self.alpha * (indices / (N - 1 + 1e-6)))
        probs   = weights / weights.sum()
        return probs

    def __get_time_weights(self):
        N = len(self.buffer)
        indices = torch.arange(N, dtype=torch.float32)

        # 給最近 N 個高權重，其他非常小的權重
        weights = torch.zeros(N)
        if N > 0:
            start = max(0, N - self.recent_n)
            weights[start:] = torch.linspace(1.0, self.alpha, N - start)  # 最近的越新越大
            weights[:start] = 1e-4  # 很老的 episode 幾乎不被抽中
        
        probs = weights / weights.sum()
        return probs

    def sample_transition(self, batch_size=1, replacement=True):
        probs   = self._get_time_weights()
        indices = torch.multinomial(probs, num_samples=batch_size, replacement=replacement)
        transitions = []
        for i in indices:
            trns = self.buffer[i.item()]["transitions"]
            transitions.append(random.choice(trns)) 
        return transitions




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
        buffer = dill.load(file)
    return buffer




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
    return  torch.ones(shape).to(device, non_blocking=True)




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
        total_loss         = loss_function(envisaged_reward[:, -1, :], desired_reward[:, -1, :])
        total_loss.backward() 

        future_action     -= future_action_.grad * (1 - future_action_ * future_action_) * beta 

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





def find_optimal_batch_size(model, dataset, device='cuda:0', bs_list=None, max_mem_ratio=0.9):

    if bs_list is None:
        bs_list = [64, 128, 256, 512, 1024]

    torch.cuda.set_device(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    results = []

    for bs in bs_list:
        torch.cuda.empty_cache(); gc.collect()
        loader = DataLoader(dataset, batch_size=bs, shuffle=False)
        batch = next(iter(loader))
        try:
            batch = [x.to(device) for x in batch]
            hs, ha, ps, fa, fr, fs = batch
            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            start = time.time()
            with torch.no_grad():
                _ = model.forward_(hs, ha, ps, fs, fa)  
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

    for history_state, history_action, present_state, future_action, future_reward, future_state in data_loader:

        history_state  = history_state .to(device)
        history_action = history_action.to(device)
        present_state  = present_state .to(device)
        future_action  = future_action .to(device)
        future_reward  = future_reward .to(device)
        future_state   = future_state  .to(device)

        model.train()
        loss_function                 = model.loss_function_
        envisaged_reward, \
        envisaged_state               = model.forward_(history_state, history_action, present_state, future_state, future_action)
        total_loss                    = torch.sum(torch.abs(loss_function(envisaged_reward, future_reward) ), dim=(1, 2)) + torch.sum(torch.abs(loss_function(envisaged_state, future_state) ), dim=(1, 2))
        TD_error_list.append(total_loss.detach())  

    TD_error = torch.cat(TD_error_list, dim=0).to(device)

    return TD_error

def update_model_per(itrtn_for_learning,
                     dataset,
                     model,
                     td_error_batch,
                     PER_exponent):
    
    device         = next(model.parameters()).device
    td_error_batch = td_error_batch
    PER_epsilon    = 1e-10
    PER_exponent   = PER_exponent

    for _ in range(itrtn_for_learning):

        obsolute_TD_error    = obtain_obsolute_TD_error(model, dataset, td_error_batch, device)
        priority             = obsolute_TD_error + PER_epsilon
        exponent_priority    = priority ** PER_exponent
        priority_probability = exponent_priority / torch.sum(exponent_priority)
        final_indices        = torch.multinomial(priority_probability, 1, replacement=False)

        batch_samples  = [dataset[i] for i in final_indices.cpu().tolist()]
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        selected_optimizer.step() 

    return model




def update_model(itrtn_for_learning,
                 buffer,
                 model,
                 batch_size):
    
    device = next(model.parameters()).device

    for _ in range(itrtn_for_learning):

        batch = buffer.sample_transition(batch_size=batch_size)

        history_state  = torch.stack([t["history_state"]  for t in batch]).to(device)
        history_action = torch.stack([t["history_action"] for t in batch]).to(device)
        present_state  = torch.stack([t["present_state"]  for t in batch]).to(device)
        future_action  = torch.stack([t["future_action"]  for t in batch]).to(device)
        future_reward  = torch.stack([t["future_reward"]  for t in batch]).to(device)
        future_state   = torch.stack([t["future_state"]   for t in batch]).to(device)
 
        model.train()
        selected_optimizer = model.selected_optimizer
        selected_optimizer.zero_grad()

        loss_function               = model.loss_function
        envisaged_reward, \
        envisaged_state             = model.forward_(history_state, history_action, present_state, future_state, future_action)
        total_loss                  = loss_function(envisaged_reward, future_reward) + loss_function(envisaged_state, future_state )
        total_loss.backward()   

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        selected_optimizer.step() 

    return model




def update_model_list(itrtn_for_learning,
                      buffer,
                      model_list,
                      param,
                      PER):
    if not PER:
        for i, model in enumerate(tqdm(model_list, desc="Updating models")):
            model_list[i] = update_model(itrtn_for_learning,
                                         buffer,
                                         model,
                                         param)
    else:
        device = next(model_list[0].parameters()).device
        td_error_batch = find_optimal_batch_size(model_list[0], buffer, device=device)
        for i, model in enumerate(tqdm(model_list, desc="Updating models")):
            model_list[i] = update_model_per(itrtn_for_learning,
                                            buffer,
                                            model,
                                            td_error_batch,
                                            param)
    return model_list




def save_performance_to_csv(performance_log, filename='performance_log.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Summed_Reward'])
        writer.writerows(performance_log)




def save_buffer_to_pickle(filename, buffer):
    with open(filename, 'wb') as file:
        dill.dump(buffer, file)


