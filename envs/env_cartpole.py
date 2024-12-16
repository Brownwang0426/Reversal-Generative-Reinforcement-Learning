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

def quantifying(start_value, end_value, tesnor_size, min_value, max_value, value):
    tensor   = torch.zeros(tesnor_size) + start_value
    interval = (max_value - min_value) / tesnor_size
    index    = int( (value - min_value) // interval + 1)
    if index >= 0:
        tensor[ : index] = end_value
    return tensor

def vectorizing_state(state, device):      # Reminder: change this for your specific task ⚠️⚠️⚠️
    state_0 = quantifying(-1, 1, 100, -4.8  , 4.8   , state[0]).to(device)
    state_1 = quantifying(-1, 1, 100, -3.75 , 3.75  , state[1]).to(device)
    state_2 = quantifying(-1, 1, 100, -0.418, 0.418 , state[2]).to(device)
    state_3 = quantifying(-1, 1, 100, -3.75 , 3.75  , state[3]).to(device)
    state   = torch.cat((state_0, state_1, state_2, state_3), dim = 0)
    return state

def vectorizing_action(pre_activated_actions, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size       = pre_activated_actions.size(2)
    action_argmax     = int(torch.argmax(pre_activated_actions[0, 0]))
    vectorized_action = torch.eye(action_size)[action_argmax].to(device)
    return vectorized_action, action_argmax

def vectorizing_reward(state, reward, summed_reward, done, reward_size, device):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    if done:
        reward = torch.zeros(reward_size).to(device)
    else:
        reward = torch.ones(reward_size).to(device)
    return reward
