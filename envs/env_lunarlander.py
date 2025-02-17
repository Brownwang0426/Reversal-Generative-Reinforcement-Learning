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

def vectorizing_state(state, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    state_0 = quantifying(-1, 1, 100, -2.5 , 2.5  , state[0], device) 
    state_1 = quantifying(-1, 1, 100, -2.5 , 2.5  , state[1], device) 
    state_2 = quantifying(-1, 1, 100, -10  , 10   , state[2], device) 
    state_3 = quantifying(-1, 1, 100, -10  , 10   , state[3], device) 
    state_4 = quantifying(-1, 1, 100, -6.28, 6.28 , state[4], device)   
    state_5 = quantifying(-1, 1, 100, -10  , 10   , state[5], device)   
    state_6 = quantifying(-1, 1, 100, 0    , 1    , state[6], device)    
    state_7 = quantifying(-1, 1, 100, 0    , 1    , state[7], device)    
    state   = torch.cat((state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7), dim = 0)
    return state

def vectorizing_action(pre_activated_actions, device):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size       = pre_activated_actions.size(2)
    action_argmax     = int(torch.argmax(pre_activated_actions[0, 0]))
    vectorized_action = (torch.eye(action_size)[action_argmax].to(device)) - 0.5 * 2
    return vectorized_action, action_argmax

def vectorizing_reward(state, reward, summed_reward, done, reward_size, device):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    reward = quantifying(-1, 1, reward_size, -200, 325, reward, device)       
    return reward


