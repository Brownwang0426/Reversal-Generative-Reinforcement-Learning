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

def vectorizing_state(state, done, device):      # Reminder: change this for your specific task ⚠️⚠️⚠️
    state_0 = quantifying(-1, 1, 100,      -1,      1 , state[0], device)
    state_1 = quantifying(-1, 1, 100,      -1,      1 , state[1], device)
    state_2 = quantifying(-1, 1, 100,      -1,      1 , state[2], device)
    state_3 = quantifying(-1, 1, 100,      -1,      1 , state[3], device)
    state_4 = quantifying(-1, 1, 100, -12.567, 12.567 , state[4], device)
    state_5 = quantifying(-1, 1, 100, -28.274, 28.274 , state[5], device)
    if done:
        state_6 = torch.ones(100).to(device)
    else:
        state_6 = torch.zeros(100).to(device) - 1
    state   = torch.cat((state_0, state_1, state_2, state_3, state_4, state_5, state_6), dim = 0)
    return state

def vectorizing_action(pre_activated_actions, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size       = pre_activated_actions.size(2)
    action_argmax     = int(torch.argmax(pre_activated_actions[0, 0]))
    vectorized_action = (torch.eye(action_size)[action_argmax].to(device) - 0.5) * 2
    return vectorized_action, action_argmax

def vectorizing_reward(state, reward, summed_reward, done, reward_size, device):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    if done:
        reward = torch.ones(reward_size).to(device) 
    else:
        reward = torch.zeros(reward_size).to(device) - 1
    return reward

class randomizer(gym.Wrapper):
    def __init__(self, env,
                 angle_pos_range=(-np.pi, np.pi),
                 angular_vel_range=(-1.0, 1.0),
                 max_attempts=100):
        super().__init__(env)
        self.angle_pos_range = angle_pos_range
        self.angular_vel_range = angular_vel_range
        self.max_attempts = max_attempts

    def reset(self, **kwargs):
        for _ in range(self.max_attempts):
            obs, info = self.env.reset(**kwargs)

            # Sample raw state: [theta1, theta2, theta1_dot, theta2_dot]
            theta1 = np.random.uniform(*self.angle_pos_range)
            theta2 = np.random.uniform(*self.angle_pos_range)
            theta1_dot = np.random.uniform(*self.angular_vel_range)
            theta2_dot = np.random.uniform(*self.angular_vel_range)

            self.env.unwrapped.state = np.array([theta1, theta2, theta1_dot, theta2_dot])

            # Get updated observation
            obs = self.env.unwrapped._get_ob()

            if self.env.observation_space.contains(obs):
                return obs, info

        print("⚠️ Warning: Couldn't find valid initial state after max attempts. Using default.")
        return self.env.reset(**kwargs)