
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

"""
# Function for vectorizing
Crucial function regarding how you manipulate or shape your state, action and reward

- It's essential to choose between immediate rewards and summed rewards for training your agent. 
  If the current state doesn't encapsulate all crucial past information, using immediate rewards is advisable. 
  This approach prevents confusion caused by varying summed rewards for the same state.

- As for reward shaping, it is recommended to increase your reward upper and decrease your reward lower bound.
"""


def quantifying(array_size, init, interval, input):
    array = np.zeros(array_size)
    index = int( (input - init) // interval + 1)
    if index >= 0:
        array[ : index] = 1
    return array

def vectorizing_state(state):      # Reminder: change this for your specific task ⚠️⚠️⚠️
    state_0 = quantifying(100, -0.6, 0.012, state[0])
    state_1 = quantifying(100, -0.1, 0.002, state[1])
    state    = np.atleast_2d(np.concatenate((state_0, state_1 )))
    return state

def vectorizing_action(action_size, action_argmax):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    return np.eye(action_size)[action_argmax]

def vectorizing_reward(state, reward, summed_reward, done, reward_size):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    reward = quantifying(reward_size, -0.6, (0.6 - (-0.6))/reward_size, state[0])
    return reward
