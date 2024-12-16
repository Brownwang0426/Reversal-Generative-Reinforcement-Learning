import gym

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

def vectorizing_state(state, device):      # Reminder: change this for your specific task ⚠️⚠️⚠️
    state = (torch.eye(16)[state].to(device) - 0.5) * 2
    return state

def vectorizing_action(pre_activated_actions, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size       = pre_activated_actions.size(2)
    action_argmax     = int(torch.argmax(pre_activated_actions[0, 0]))
    vectorized_action = torch.eye(action_size)[action_argmax].to(device)
    return vectorized_action, action_argmax

def vectorizing_reward(state, reward, summed_reward, done, reward_size, device):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    if done: 
        if (state == 15):         # If the agent reaches goal
            reward = torch.ones(reward_size).to(device)
        else:
            reward = torch.zeros(reward_size).to(device)
    else:
        x, y = divmod(state, 4)
        distance = np.sqrt((x - 3) ** 2 + (y - 3) ** 2)
        max_distance = np.sqrt(3**2 + 3**2)  # 4.24
        idx = int(100 * (1 - (distance / max_distance)))
        reward = torch.zeros(reward_size).to(device)
        reward[0: idx        ] = 1
    return reward
