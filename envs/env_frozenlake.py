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

def quantifying(start_value, end_value, array_size, min_value, max_value, value):
    array    = np.zeros(array_size) + start_value
    interval = (max_value - min_value) / array_size
    index    = int( (value - min_value) // interval + 1)
    if index >= 0:
        array[ : index] = end_value
    return array

def vectorizing_state(state):      # Reminder: change this for your specific task ⚠️⚠️⚠️
    state = (np.eye(16)[state] - 0.5) * 2
    return state

def vectorizing_action(pre_activated_actions):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size      = pre_activated_actions.size(2)
    action_argmax    = int(torch.argmax(pre_activated_actions[0, 0]))
    return np.eye(action_size)[action_argmax], action_argmax

def vectorizing_reward(state, reward, summed_reward, done, reward_size):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    x, y          = divmod(state, 4)
    distance      = np.sqrt((x - 3) ** 2 + (y - 3) ** 2)
    max_distance  = np.sqrt(3**2 + 3**2)  # 4.24
    idx           = int(100 * (1 - (distance / max_distance)))
    reward        = np.zeros(reward_size)
    reward[0: idx] = 1
    return reward