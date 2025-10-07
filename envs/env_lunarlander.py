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

def vectorizing_state(state, done, truncated, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    null_state = torch.ones(10).to(device)
    if done or truncated:
        state_0 = torch.ones(50).to(device)
    else:
        state_0 = torch.zeros(50).to(device) - 1
    state_1 = quantifying(-1, 1, 50, -2.5 , 2.5  , state[0], device) 
    state_2 = quantifying(-1, 1, 50, -2.5 , 2.5  , state[1], device) 
    state_3 = quantifying(-1, 1, 50, -10  , 10   , state[2], device) 
    state_4 = quantifying(-1, 1, 50, -10  , 10   , state[3], device) 
    state_5 = quantifying(-1, 1, 50, -6.28, 6.28 , state[4], device)   
    state_6 = quantifying(-1, 1, 50, -10  , 10   , state[5], device)   
    state_7 = quantifying(-1, 1, 50, 0    , 1    , state[6], device)    
    state_8 = quantifying(-1, 1, 50, 0    , 1    , state[7], device)    
    state   = torch.cat((null_state, state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8), dim = 0)
    return state

def vectorizing_action(pre_activated_actions, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size       = pre_activated_actions.size(2) 
    action_argmax     = int(torch.argmax(pre_activated_actions[0, 0, :]))
    vectorized_action = (torch.eye(action_size)[action_argmax].to(device) - 0.5) * 2
    return vectorized_action, action_argmax 

def vectorizing_reward(state, done, truncated, reward, summed_reward, reward_size, device):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    if done or truncated: 
        if done:
            reward = quantifying(-1, 1, reward_size , -200, 325, summed_reward, device)       
        else:
            reward = quantifying(-1, 1, reward_size , -200, 325, summed_reward, device)   
    else:
        reward = quantifying(-1, 1, reward_size , -200, 325, summed_reward, device)       
    return reward

def quantized_highest_reward(performance_log, batch_size): # Reminder: change this for your specific task ⚠️⚠️⚠️
    start_value = -200
    end_value = 325   
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

class randomizer(gym.Wrapper):
    def __init__(self, env, pos_range=(-0.1, 0.1), vel_range=(-0.1, 0.1),
                 angle_pos_range=(-0.1, 0.1), angle_vel_range=(-0.1, 0.1),
                 max_attempts=100):
        super().__init__(env)
        self.pos_range = pos_range
        self.vel_range = vel_range
        self.angle_pos_range = angle_pos_range
        self.angle_vel_range = angle_vel_range
        self.max_attempts = max_attempts

        # Constants from source
        self.VIEWPORT_W = 600
        self.SCALE = 30.0

    def reset(self, **kwargs):
        for _ in range(self.max_attempts):
            obs, info = self.env.reset(**kwargs)

            lander = self.env.unwrapped.lander
            if lander is None:
                return obs, info

            # Position: world units
            x_center = self.VIEWPORT_W / self.SCALE / 2
            x_pos = x_center + np.random.uniform(*self.pos_range)
            y_pos = np.random.uniform(1.0, 1.4)

            # Random y_posnamics
            x_vel = np.random.uniform(*self.vel_range)
            y_vel = np.random.uniform(*self.vel_range)
            angle = np.random.uniform(*self.angle_pos_range)
            angle_vel = np.random.uniform(*self.angle_vel_range)

            # Set Box2D boy_pos state
            lander.position = (x_pos, y_pos)
            lander.linearVelocity = (x_vel, y_vel)
            lander.angle = angle
            lander.angularVelocity = angle_vel

            # Force observation recomputation by stepping with zero action
            obs, _, terminated, truncated, _ = self.env.step(0)

            # Check if state is reasonable
            if not np.any(np.isnan(obs)) and not terminated and not truncated:
                return obs, info

        print("⚠️ Warning: Couldn't find valid initial state after max attempts. Using default.")
        return self.env.reset(**kwargs)


class randomizer(gym.Wrapper):
    def __init__(self, env, 
                 x_pos_range=(5, 15), x_vel_range=(-1, 1),
                 y_pos_range=(3.5, 11), y_vel_range=(-1, 1),
                 angle_pos_range=(-1, 1), angle_vel_range=(-1, 1), 
                 max_attempts=50):
        super().__init__(env)
        self.x_pos_range = x_pos_range
        self.x_vel_range = x_vel_range
        self.y_pos_range = y_pos_range
        self.y_vel_range = y_vel_range
        self.angle_pos_range = angle_pos_range
        self.angle_vel_range = angle_vel_range
        self.max_attempts = max_attempts

        self.SCALE = 30
        self.VIEWPORT_W = 600 # 600/30= 20
        self.VIEWPORT_H = 400 # 400/30= 13.33

    def reset(self, **kwargs):
        for _ in range(self.max_attempts):
            obs, info = self.env.reset(**kwargs)

            lander = self.env.unwrapped.lander
            legs = self.env.unwrapped.legs
            if lander is None or legs is None:
                return obs, info

            # Random offsets from center
            x_pos = np.random.uniform(*self.x_pos_range)
            x_vel = np.random.uniform(*self.x_vel_range)
            y_pos = np.random.uniform(*self.y_pos_range)  
            y_vel = np.random.uniform(*self.y_vel_range)
            angle_pos = np.random.uniform(*self.angle_pos_range)
            angle_vel = np.random.uniform(*self.angle_vel_range)

            # Move lander
            lander.position = (x_pos, y_pos)
            lander.linearVelocity = (x_vel, y_vel)
            lander.angle = angle_pos
            lander.angularVelocity = angle_vel

            # Move legs accordingly
            for leg in legs:
                leg.position = (x_pos, y_pos)
                leg.linearVelocity = (x_vel, y_vel)
                leg.angle = angle_pos
                leg.angularVelocity = angle_vel

            # Allow Box2D to stabilize the scene
            obs, _, terminated, truncated, _ = self.env.step(0)

            if not np.any(np.isnan(obs)) and not terminated and not truncated:
                return obs, info

        print("⚠️ Warning: Couldn't find valid initial state after max attempts. Using default.")
        return self.env.reset(**kwargs)
