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
    state_1 = quantifying(-1, 1, 100, -0.6, 0.6, state[0], device)
    state_2 = quantifying(-1, 1, 100, -0.1, 0.1, state[1], device)
    state   = torch.cat((null_state, state_0, state_1, state_2), dim = 0)
    return state

def vectorizing_action(pre_activated_actions, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size       = pre_activated_actions.size(2) 
    action_argmax     = int(torch.argmax(pre_activated_actions[0, 0, :]))
    vectorized_action = (torch.eye(action_size)[action_argmax].to(device, non_blocking=True) - 0.5) * 2
    return vectorized_action, action_argmax 

def vectorizing_reward(state, done, truncated, reward, summed_reward, reward_size, device):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    if done or truncated: 
        if state[0]>=0.5:
            reward = quantifying(-1, 1, reward_size , -1.2, 0.6, state[0], device)
        else:
            reward = quantifying(-1, 1, reward_size , -1.2, 0.6, state[0], device)
    else:
        reward = quantifying(-1, 1, reward_size , -1.2, 0.6, state[0], device)
    return reward

def itrtn_by_averaging_reward(performance_log, itrtn_for_planning, window_size): # Reminder: change this for your specific task ⚠️⚠️⚠️
    start_value = -200
    end_value = -90   
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
    def __init__(self, env,
                 pos_range=(-1.5, 0.5),
                 vel_range=(-0.05, 0.05),
                 max_attempts=100):
        super().__init__(env)
        self.pos_range = pos_range
        self.vel_range = vel_range
        self.max_attempts = max_attempts

    def reset(self, **kwargs):
        for _ in range(self.max_attempts):
            obs, info = self.env.reset(**kwargs)

            position = np.random.uniform(*self.pos_range)
            velocity = np.random.uniform(*self.vel_range)

            self.env.unwrapped.state = np.array([position, velocity])
            obs = np.array([position, velocity], dtype=np.float32)

            if self.env.observation_space.contains(obs):
                return obs, info

        print("⚠️ Warning: Couldn't find valid initial state after max attempts. Using default.")
        return self.env.reset(**kwargs)

