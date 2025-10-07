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
    tensor   = torch.zeros(tesnor_size).to(device) + start_value
    interval = (max_value - min_value) / tesnor_size
    index    = int( (value - min_value) // interval + 1)
    if index >= 0:
        tensor[ : index] = end_value
    return tensor

def vectorizing_state(state, done, truncated, device):      # Reminder: change this for your specific task ⚠️⚠️⚠️
    null_state = torch.ones(10).to(device)
    if done or truncated:
        state_0 = torch.ones(100).to(device)
    else:
        state_0 = torch.zeros(100).to(device) - 1
    state_1 = quantifying(-1, 1, 10,  1 , 4, state['direction'], device)
    state_2 = (torch.tensor(state['image'].ravel()/10).to(device) - 0.5) * 2
    state   = torch.cat((null_state, state_0.float(), state_1.float(), state_2.float()), dim = 0)
    return state

def vectorizing_action(pre_activated_actions, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size       = pre_activated_actions.size(2) 
    action_argmax     = int(torch.argmax(pre_activated_actions[0, 0, :]))
    vectorized_action = (torch.eye(action_size)[action_argmax].to(device) - 0.5) * 2
    return vectorized_action, action_argmax 

def quantized_highest_reward(performance_log, batch_size): # Reminder: change this for your specific task ⚠️⚠️⚠️
    start_value = 0
    end_value = 1 
    N = 50 
    recent_K = batch_size * 2
    rewards = []
    if performance_log:
        for item in performance_log[-recent_K:]:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                _, r = item
            else:
                r = item
            rewards.append(r)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    scaled = (avg_reward - start_value) / (end_value - start_value)
    scaled = max(0.0, min(1.0, scaled))  # clamp 0~1
    iteration_unit = int(N * scaled) 
    return iteration_unit

def vectorizing_reward(state, done, truncated, reward, summed_reward, reward_size, device):     # Reminder: change this for your specific task ⚠️⚠️⚠️
    if done or truncated: 
        if done:
            reward = quantifying(-1, 1, reward_size, 0, 1, reward, device)      
        else:
            reward = quantifying(-1, 1, reward_size, 0, 1, reward, device)    
    else:
        reward = quantifying(-1, 1, reward_size , 0, 1, reward, device)
    return reward




from minigrid.core.world_object import Door, Key

class randomizer(gym.Wrapper):
    def __init__(self, env, max_attempts=100):
        super().__init__(env)
        self.max_attempts = max_attempts

    def reset(self, **kwargs):
        
        obs, info = self.env.reset(**kwargs)
        env = self.env.unwrapped
        grid = env.grid

        width, height = grid.width, grid.height

        for _ in range(self.max_attempts):

            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            if grid.get(x, y) is None or grid.get(x, y).can_overlap():
                env.agent_pos = (x, y)
                env.agent_dir = np.random.choice(4)

                doors = []
                keys = []
                for i in range(width):
                    for j in range(height):
                        cell = grid.get(i, j)
                        if isinstance(cell, Door):
                            doors.append(cell)
                        elif isinstance(cell, Key):
                            keys.append(cell)

                for door in doors:
                    door.is_open = np.random.choice([True, False])

                for key in keys:
                    if np.random.rand() < 0.5:
                        grid.set(*key.cur_pos, None)  

                obs = env.gen_obs()
                return obs, info

        print("⚠️ Warning: Couldn't find valid initial state after max attempts. Using default.")
        return self.env.reset(**kwargs)
