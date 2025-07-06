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
    state_0 = quantifying(-1, 1, 50, -0.6, 0.6, state[0], device)
    state_1 = quantifying(-1, 1, 50, -0.1, 0.1, state[1], device)
    if done or truncated:
        state_2 = torch.ones(50).to(device)
    else:
        state_2 = torch.zeros(50).to(device) - 1
    state   = torch.cat((null_state, state_0, state_1, state_2), dim = 0)
    return state

def vectorizing_action(pre_activated_actions, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size       = pre_activated_actions.size(2) 
    action_argmax     = int(torch.argmax(pre_activated_actions[0, 0, 1:]))
    vectorized_action = (torch.eye(action_size)[action_argmax + 1 ].to(device) - 0.5) * 2
    vectorized_action[0] = 1
    return vectorized_action, action_argmax 

def vectorizing_reward(state, done, truncated, reward, summed_reward, reward_size, device):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    null_reward = torch.ones(10).to(device)
    if done or truncated: 
        if state[0]>=0.5:
            reward = quantifying(-1, 1, reward_size - 10, -1.2, 0.6, state[0], device)
        else:
            reward = torch.zeros(reward_size - 10).to(device) - 1
    else:
        reward = quantifying(-1, 1, reward_size - 10, -1.2, 0.6, state[0], device)
    reward = torch.cat((null_reward, reward), dim = 0)
    return reward

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

