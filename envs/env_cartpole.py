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
    if done:
        state_0 = torch.ones(50).to(device)
    else:
        state_0 = torch.zeros(50).to(device) - 1
    state_1 = quantifying(-1, 1, 50, -4.8  , 4.8   , state[0], device)
    state_2 = quantifying(-1, 1, 50, -3.75 , 3.75  , state[1], device)
    state_3 = quantifying(-1, 1, 50, -0.418, 0.418 , state[2], device)
    state_4 = quantifying(-1, 1, 50, -3.75 , 3.75  , state[3], device)
    state   = torch.cat((null_state, state_0, state_1, state_2, state_3, state_4), dim = 0)
    return state

def vectorizing_action(pre_activated_actions, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size       = pre_activated_actions.size(2) 
    action_argmax     = int(torch.argmax(pre_activated_actions[0, 0, :]))
    vectorized_action = (torch.eye(action_size)[action_argmax].to(device) - 0.5) * 2
    return vectorized_action, action_argmax 

def vectorizing_reward(state, done, truncated, reward, summed_reward, reward_size, device):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    reward = quantifying(-1, 1, reward_size, 0, 1, reward, device)
    return reward

class randomizer(gym.Wrapper):
    def __init__(self, env, 
                 pos_range=(-1.5, 1.5), 
                 vel_range=(-1, 1),
                 angle_pos_range=(-1, 1), 
                 angle_vel_range=(-1, 1),
                 max_attempts=100):
        super().__init__(env)
        self.pos_range = pos_range
        self.vel_range = vel_range
        self.angle_pos_range = angle_pos_range
        self.angle_vel_range = angle_vel_range
        self.max_attempts = max_attempts

    def reset(self, **kwargs):
        def is_done(state):
            x, x_dot, theta, theta_dot = state
            x_threshold = 2.4
            theta_threshold_radians = 12 * 2 * np.pi / 360  
            return (x < -x_threshold or x > x_threshold or
                    theta < -theta_threshold_radians or theta > theta_threshold_radians)

        for _ in range(self.max_attempts):
            obs, info = self.env.reset(**kwargs)

            state = np.array([
                np.random.uniform(*self.pos_range),
                np.random.uniform(*self.vel_range),
                np.random.uniform(*self.angle_pos_range),
                np.random.uniform(*self.angle_vel_range)
            ])
            self.env.unwrapped.state = state
            obs = state

            if not is_done(state):
                return obs, info

        print("⚠️ Warning: Couldn't find valid initial state after max attempts. Using default.")
        return self.env.reset(**kwargs)