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

"""
# Function for vectorizing
Crucial function regarding how you manipulate or shape your state, action and reward

- It's essential to choose between immediate rewards and summed rewards for training your agent. 
  If the current state doesn't encapsulate all crucial past information, using immediate rewards is advisable. 
  This approach prevents confusion caused by varying summed rewards for the same state.

- As for reward shaping, it is recommended to increase your reward upper and decrease your reward lower bound.
"""

def quantifying(start_value, end_value, tesnor_size, min_value, max_value, value, device):
    tensor   = torch.zeros(tesnor_size).to(device, non_blocking=True) + start_value
    interval = (max_value - min_value) / tesnor_size
    index    = int( (value - min_value) // interval + 1)
    if index >= 0:
        tensor[ : index] = end_value
    return tensor

def vectorizing_state(state, done, truncated, device):      # Reminder: change this for your specific task ⚠️⚠️⚠️
    null_state = torch.ones(10).to(device, non_blocking=True)
    if done or truncated:
        state_0 = torch.ones(10).to(device, non_blocking=True)
    else:
        state_0 = torch.zeros(10).to(device, non_blocking=True) - 1
    state_1 = (torch.eye(16)[state].to(device, non_blocking=True) - 0.5) * 2
    state   = torch.cat((null_state, state_0, state_1), dim = 0)
    return state

def vectorizing_action(pre_activated_actions, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size       = pre_activated_actions.size(2) 
    action_argmax     = int(torch.argmax(pre_activated_actions[0, 0, :]))
    vectorized_action = (torch.eye(action_size)[action_argmax].to(device, non_blocking=True) - 0.5) * 2
    return vectorized_action, action_argmax 

def vectorizing_reward(state, done, truncated, reward, summed_reward, reward_size, device):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    if done or truncated: 
        if done:         # If the agent reaches goal
            reward = quantifying(-1, 1, reward_size , 0, 1, summed_reward, device)      
        else:
            reward = quantifying(-1, 1, reward_size , 0, 1, summed_reward, device)      
            # x, y = divmod(state, 4)
            # distance = np.sqrt((x - 3) ** 2 + (y - 3) ** 2)
            # max_distance = np.sqrt(3**2 + 3**2)  # 4.24
            # idx = int(100 * (1 - (distance / max_distance)))
            # reward = torch.zeros(reward_size ).to(device, non_blocking=True) - 1
            # reward[0: idx] = 1
    else:
        reward = quantifying(-1, 1, reward_size , 0, 1, summed_reward, device)      
        # x, y = divmod(state, 4)
        # distance = np.sqrt((x - 3) ** 2 + (y - 3) ** 2)
        # max_distance = np.sqrt(3**2 + 3**2)  # 4.24
        # idx = int(100 * (1 - (distance / max_distance)))
        # reward = torch.zeros(reward_size ).to(device, non_blocking=True) - 1
        # reward[0: idx] = 1
    return reward

def itrtn_by_averaging_reward(performance_log, itrtn_for_planning, window_size): # Reminder: change this for your specific task ⚠️⚠️⚠️
    start_value = 0 
    end_value = 1   
    N = itrtn_for_planning
    recent_K = window_size
    rewards = []
    if performance_log:
        for item in performance_log[-recent_K:]:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                _, r = item
            else:
                r = item
            rewards.append(r)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    scaled = (np.clip(avg_reward, start_value, end_value) - start_value) / (end_value - start_value)
    scaled = np.clip(scaled, 0.0, 1.0)
    iteration_unit = int(N * scaled) 
    return iteration_unit

class randomizer(gym.Wrapper):
    def __init__(self, env, max_attempts=100):
        super().__init__(env)
        self.max_attempts = max_attempts
        self.desc = env.unwrapped.desc.astype('U') 
        self.valid_start_positions = [
            i for i, c in enumerate(self.desc.flatten())
            if c in ['F', 'S'] 
        ]

    def reset(self, **kwargs):
        for _ in range(self.max_attempts):

            start_state = np.random.choice(self.valid_start_positions)

            obs, info = self.env.reset(**kwargs)
            self.env.unwrapped.s = start_state  
            obs = start_state

            done = (obs == self.env.unwrapped.nrow * self.env.unwrapped.ncol - 1)  
            if not done:
                return obs, info

        print("⚠️ Warning: Couldn't find valid initial state after max attempts. Using default.")
        return self.env.reset(**kwargs)
