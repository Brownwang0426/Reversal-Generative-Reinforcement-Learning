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

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    device_index = 0
    device = torch.device(f"cuda:{device_index}")
    print('using cuda...')
else:
    device = torch.device("cpu")
    print('using cpu...')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True







game_name =  'FrozenLake-v1'         #⚠️   gym.make(game_name, max_episode_steps=max_steps_for_each_episode, is_slippery=False, map_name="4x4")
max_steps_for_each_episode = 20      #⚠️
seed = None                          #⚠️

load_pretrained_model = True

ensemble_size = 5                    #◀️

validation_size = 10                 #◀️

state_size = 36                      #⚠️
action_size = 4                      #⚠️
reward_size = 100                    #⚠️
feature_size = 200                   #⚠️
history_size = 0                     #⚠️
future_size = 5                      #⚠️
neural_type = 'td_mini'              #⚠️
num_layers = 3                       #⚠️
num_heads = 10                       #⚠️


init = "xavier_normal"
opti = 'sgd'
loss = 'mean_squared_error'
bias = False
drop_rate = 0.
alpha = 0.1                  
itrtn_for_learning  = 100

beta = 0.1                     
itrtn_for_planning  = 20     

episode_for_training = 100000

batch_size_for_executing = 1         #⚠️

batch_size_for_learning = 1          #⚠️       

buffer_limit = 10000   






episode_for_testing = 100
render_for_human = True

suffix                 = f"game_{game_name}-type_{neural_type}-ensemble_{ensemble_size:05d}-learn_{itrtn_for_learning:05d}-plan_{itrtn_for_planning:05d}"
directory              = f'./result/{game_name}/'
performance_directory  = f'./result/{game_name}/performace-{suffix}.csv'
model_directory        = f'./result/{game_name}/model-{suffix}.pth'
buffer_directory       = f'./result/{game_name}/buffer-{suffix}.dill'

if not os.path.exists(directory):
    os.makedirs(directory)


game_modules = {
    'FrozenLake-v1': 'envs.env_frozenlake',
    'CartPole-v1': 'envs.env_cartpole',
    'MountainCar-v0': 'envs.env_mountaincar',
    'Acrobot-v1': 'envs.env_acrobot',
    'LunarLander-v3': 'envs.env_lunarlander',
    'MiniGrid-DoorKey-5x5-v0': 'envs.env_doorkey'
}
if game_name in game_modules:
    game_module = __import__(game_modules[game_name], fromlist=['vectorizing_state', 'vectorizing_action', 'vectorizing_reward', 'randomizer'])
    vectorizing_state  = game_module.vectorizing_state
    vectorizing_action = game_module.vectorizing_action
    vectorizing_reward = game_module.vectorizing_reward
    randomizer         = game_module.randomizer
else:
    raise RuntimeError('Missing env functions')




model_modules = {
    'td_mini_linear': 'models.model_td_mini_linear',
    'td_mini': 'models.model_td_mini',
    'rnn_mini': 'models.model_rnn_mini',
    'gru_mini': 'models.model_rnn_mini',
    'lstm_mini': 'models.model_rnn_mini',
    'td': 'models.model_td',
    'rnn': 'models.model_rnn',
    'gru': 'models.model_rnn',
    'lstm': 'models.model_rnn',
    'td_rnn': 'models.model_td_rnn'
}
if neural_type in model_modules:
    model_module = __import__(model_modules[neural_type], fromlist=['build_model'])
    build_model  = model_module.build_model
else:
    raise RuntimeError('Missing model functions')

from utils.util_func  import load_performance_from_csv,\
                             load_buffer_from_pickle,\
                             retrieve_history,\
                             retrieve_present,\
                             initialize_future_action, \
                             initialize_desired_reward,\
                             update_future_action, \
                             sequentialize, \
                             update_long_term_experience_replay_buffer,\
                             update_model_list,\
                             limit_buffer,\
                             save_performance_to_csv,\
                             save_buffer_to_pickle




# creating empty log for recording performance
performance_log  = []

# setting the last episode number for performance log
last_episode = 0

# creating model list
sequence_size = history_size + future_size 
model_list = []
for _ in range(ensemble_size):
    model = build_model(state_size,
                        action_size,
                        reward_size,
                        feature_size,
                        sequence_size,
                        neural_type,
                        num_layers,
                        num_heads,
                        init,
                        opti,
                        loss,
                        bias,
                        drop_rate,
                        alpha)
    model.to(device)
    model_list.append(model)

# creating space for storing tensors as experience replay buffer
history_state_stack        = torch.empty(0).to(device)
history_action_stack       = torch.empty(0).to(device)
present_state_stack        = torch.empty(0).to(device)
future_action_stack        = torch.empty(0).to(device)
future_reward_stack        = torch.empty(0).to(device)
future_state_stack         = torch.empty(0).to(device)
history_state_hash_list    = list()
history_action_hash_list   = list()
present_state_hash_list    = list()
future_action_hash_list    = list()
future_reward_hash_list    = list()
future_state_hash_list     = list()

# load from pre-trained models if needed
if load_pretrained_model == True:
    try:
        model_dict = torch.load(model_directory)
        for i, model in enumerate(model_list):
            model.load_state_dict(model_dict[f'model_{i}'])
        history_state_stack, \
        history_action_stack,\
        present_state_stack, \
        future_action_stack, \
        future_reward_stack, \
        future_state_stack,  \
        history_state_hash_list, \
        history_action_hash_list, \
        present_state_hash_list, \
        future_action_hash_list, \
        future_reward_hash_list, \
        future_state_hash_list = load_buffer_from_pickle(buffer_directory)
        history_state_stack    = history_state_stack.to (device) 
        history_action_stack   = history_action_stack.to(device) 
        present_state_stack    = present_state_stack.to (device) 
        future_action_stack    = future_action_stack.to (device) 
        future_reward_stack    = future_reward_stack.to (device) 
        future_state_stack     = future_state_stack .to (device) 
        performance_log        = load_performance_from_csv(performance_directory)
        last_episode           = performance_log[-1][0] if len(performance_log) > 0 else 0
        print('Loaded pre-trained models.')
    except:
        print('Failed loading pre-trained models. Now using new models.')






# starting each episode
for training_episode in tqdm(range(episode_for_training)):
    current_episode  = training_episode + last_episode + 1
    
    # initializing summed reward
    summed_reward  = 0

    # initializing short term experience replay buffer
    state_list  = []
    action_list = []
    reward_list = []
    for _ in range(history_size):
        state_list .append(torch.zeros(state_size  ).to(device) - 1 )
        action_list.append(torch.zeros(action_size ).to(device) - 1 )
        reward_list.append(torch.zeros(reward_size ).to(device) - 1 ) 

    # initializing environment
    if game_name == 'FrozenLake-v1'  :
        env        = gym.make(game_name, max_episode_steps=max_steps_for_each_episode, is_slippery=False, map_name="4x4")
    else:
        env        = gym.make(game_name, max_episode_steps=max_steps_for_each_episode)
    state, info    = env.reset(seed = seed)
    
    # observing state
    state          = vectorizing_state(state, False, False, device)
    state_list.append(state)

    # starting each step
    post_done_truncated_counter = 0
    post_done_truncated_steps = future_size
    done_truncated_flag = False
    total_step = 0 
    while not done_truncated_flag:
        
        """
        We let agent took some history states and actions into consideration.
        """
        """
        The final desired reward is factually the last time step in desired reward.
        """
        # initializing and updating action by desired reward                                  
        history_state, \
        history_action  = retrieve_history(state_list, action_list, history_size, device)
        present_state   = retrieve_present(state_list, device)
        future_action   = initialize_future_action((1, future_size, action_size), device)
        desired_reward  = initialize_desired_reward((1, future_size, reward_size), device)
        future_action   = update_future_action(np.random.randint(itrtn_for_planning) + 1,
                                               model_list,
                                               history_state ,
                                               history_action,
                                               present_state,
                                               future_action,
                                               desired_reward,
                                               beta)

        """
        We let agent execute several planned actions rather than one at a time to make data gathering more efficient. 
        batch_size_for_executing shall be less or equal to future_size.
        """
        # taking actions and skip planning 
        for i in range(batch_size_for_executing):

            # observing action
            action, action_  = vectorizing_action(future_action[:, i:, :], device)
            action_list.append(action)

            # executing action
            state, reward, done, truncated, info = env.step(action_)

            # summing reward
            if post_done_truncated_counter > 0:
                reward = 0
            summed_reward += reward

            # observing actual reward
            reward = vectorizing_reward(state, done, truncated, reward, summed_reward, reward_size, device)
            reward_list.append(reward)

            # observing state
            state = vectorizing_state(state, done, truncated, device)
            state_list.append(state)

            """
            We expanded the condition for terminating an episode to include the case where the count is smaller than the sum of the history and future sizes. 
            Though it is contrary to common practice in RL, this is for better handling the sequentialization of the short-term experience replay buffer with fixed window length.
            And it is also for agent to plan ahead even after the episode is done.
            We give a done flag to state to indicate that the environment is done so that the agent won't be confused. 
            The done flag shall affect the state in a considerable way to remind the agent that the environment is done.
            """
            # if done then continue for a short period. Then store experience to short term experience replay buffer
            if done or truncated:
                post_done_truncated_counter += 1
                if post_done_truncated_counter >= post_done_truncated_steps:
                    done_truncated_flag = True
                    break            
            else:
                total_step += 1
                print(f'\rStep: {total_step}\r', end='', flush=True)
                
    # closing env
    env.close()




    # recording performance
    print(f'Episode {current_episode}: Summed_Reward = {summed_reward}')
    performance_log.append([current_episode, summed_reward])




    # sequentializing short term experience replay buffer
    history_state_list   ,\
    history_action_list   ,\
    present_state_list   ,\
    future_action_list   ,\
    future_reward_list   ,\
    future_state_list    = sequentialize(state_list  ,
                                         action_list ,
                                         reward_list ,
                                         history_size,
                                         future_size)


    

    # storing sequentialized short term experience to long term experience replay buffer 
    history_state_stack, \
    history_action_stack, \
    present_state_stack, \
    future_action_stack, \
    future_reward_stack, \
    future_state_stack,\
    history_state_hash_list  , \
    history_action_hash_list  , \
    present_state_hash_list  , \
    future_action_hash_list  , \
    future_reward_hash_list  , \
    future_state_hash_list      = update_long_term_experience_replay_buffer(history_state_stack,
                                                                            history_action_stack,
                                                                            present_state_stack,
                                                                            future_action_stack,
                                                                            future_reward_stack,
                                                                            future_state_stack ,
                                                                            history_state_hash_list  ,
                                                                            history_action_hash_list  ,
                                                                            present_state_hash_list  ,
                                                                            future_action_hash_list  ,
                                                                            future_reward_hash_list  ,
                                                                            future_state_hash_list   ,
                                                                            history_state_list   ,
                                                                            history_action_list   ,
                                                                            present_state_list,
                                                                            future_action_list,
                                                                            future_reward_list,
                                                                            future_state_list )


    

    """
    We use batch_size to make training more efficient.
    """
    """
    We can also use prioritized experience replay to make training more efficient.
    """
    # training
    if current_episode % validation_size == 0:
        dataset     = TensorDataset    (history_state_stack,
                                        history_action_stack,
                                        present_state_stack,
                                        future_action_stack,
                                        future_reward_stack,
                                        future_state_stack  )
        model_list  = update_model_list (itrtn_for_learning ,
                                        dataset,
                                        model_list,
                                        batch_size_for_learning
                                        )





        # limit_buffer
        history_state_stack, \
        history_action_stack, \
        present_state_stack, \
        future_action_stack, \
        future_reward_stack, \
        future_state_stack , \
        history_state_hash_list  , \
        history_action_hash_list  , \
        present_state_hash_list  , \
        future_action_hash_list  , \
        future_reward_hash_list  , \
        future_state_hash_list   = limit_buffer(history_state_stack,
                                                history_action_stack,
                                                present_state_stack,
                                                future_action_stack,
                                                future_reward_stack,
                                                future_state_stack ,
                                                history_state_hash_list  ,
                                                history_action_hash_list  ,
                                                present_state_hash_list  ,
                                                future_action_hash_list  ,
                                                future_reward_hash_list  ,
                                                future_state_hash_list ,
                                                buffer_limit  )




        # saving nn models
        model_dict = {}
        for i, model in enumerate(model_list):
            model_dict[f'model_{i}'] = model.state_dict()
        torch.save(model_dict, model_directory)

        # saving long term experience replay buffer
        save_buffer_to_pickle(buffer_directory,
                              history_state_stack,
                              history_action_stack,
                              present_state_stack,
                              future_action_stack,
                              future_reward_stack,
                              future_state_stack,
                              history_state_hash_list,
                              history_action_hash_list,
                              present_state_hash_list,
                              future_action_hash_list,
                              future_reward_hash_list,
                              future_state_hash_list)

        # saving final reward to log
        save_performance_to_csv(performance_log, performance_directory)




        # clear up
        gc.collect()
        torch.cuda.empty_cache()