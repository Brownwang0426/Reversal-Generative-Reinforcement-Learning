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
        for episode in self.buffer:
            for transition in episode["transitions"]:
                for key, value in transition.items():
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
        logits = self.alpha * (indices / (N - 1 + 1e-6))
        probs = torch.softmax(logits, dim=0)
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




def update_model(itrtn_for_learning,
                 buffer,
                 model,
                 param):
    
    device = next(model.parameters()).device

    for _ in range(itrtn_for_learning):

        buffer.alpha = param
        batch = buffer.sample_transition(batch_size=1)
        
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
                      param):
    for i, model in enumerate(tqdm(model_list, desc="Updating models")):
        model_list[i] = update_model(itrtn_for_learning,
                                        buffer,
                                        model,
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


