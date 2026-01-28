
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




class custom_attn(nn.Module):
    def __init__(self, feature_size, num_heads, bias, drop_rate):
        super(custom_attn, self).__init__()
        assert feature_size % num_heads == 0, "feature_size must be divisible by num_heads"
        self.feature_size  = feature_size
        self.num_heads     = num_heads
        self.head_size     = feature_size // num_heads
        self.bias          = bias
        self.drop_rate     = drop_rate
        self.W_q           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_k           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_v           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_o           = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.attn_dropout  = nn.Dropout(self.drop_rate)

    def split_heads(self, x):
        batch_size, sequence_size, feature_size = x.size()
        return x.view(batch_size, sequence_size, self.num_heads, self.head_size).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask):

        # attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5) #  (batch_size, num_heads, sequence_size, head_size) @ (batch_size, num_heads, head_size, sequence_size ) 
        K_T = K.transpose(-2, -1).contiguous()
        attn_scores = (Q @ K_T) / (self.head_size ** 0.5)

        if mask != None:
            attn_scores = attn_scores + mask                   # (batch_size, num_heads, sequence_size, sequence_size) += (batch_size, 1, sequence_size, sequence_size)
        else:
            pass

        attn_probs = torch.softmax(attn_scores, dim=-1) 
        attn_probs = self.attn_dropout (attn_probs)
        # output     = torch.matmul(attn_probs, V)  # (batch_size, num_heads, sequence_size, sequence_size) @ (batch_size, num_heads, sequence_size, head_size ) 
        output     = attn_probs @ V
        return output                               # (batch_size, num_heads, sequence_size, head_size)

    def combine_heads(self, x):
        batch_size, num_heads, sequence_size, head_size = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_size, self.feature_size)

    def forward(self, Q, K, V, mask=None, kv_cache=None):
        # mask Shape: (batch_size, 1, sequence_size, sequence_size)
        # Q    Shape: (batch_size,    sequence_size, feature_size )
        Q    = self.split_heads(self.W_q(Q))  # Shape: (batch_size, num_heads, sequence_size, head_size )
        K    = self.split_heads(self.W_k(K))  # Shape: (batch_size, num_heads, sequence_size, head_size )
        V    = self.split_heads(self.W_v(V))  # Shape: (batch_size, num_heads, sequence_size, head_size )
        if kv_cache is not None:
            if 'k' in kv_cache and 'v' in kv_cache:
                K = torch.cat([kv_cache['k'], K], dim=2)
                V = torch.cat([kv_cache['v'], V], dim=2)
            kv_cache['k'] = K
            kv_cache['v'] = V
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output      = self.W_o(self.combine_heads(attn_output))
        return output, kv_cache




class moe_ffn(nn.Module):
    def __init__(self, feature_size, num_experts=4, top_k=2, bias=False):
        super(moe_ffn, self).__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.bias        = bias
        self.experts     = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_size, feature_size, bias=self.bias),
                nn.GELU(),
                nn.Linear(feature_size, feature_size, bias=self.bias)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(feature_size, num_experts)

    def forward(self, x):
        
        # get shape
        B, T, D = x.shape

        # caculate gate scores
        gate_scores        = self.gate(x)      # [B, T, D] -> [B, T, num_experts]
        topk_val, topk_idx = torch.topk(gate_scores, self.top_k, dim=-1)  # [B, T, top_k]
    
        # build weights
        weights = F.softmax(topk_val, dim=-1)  # [B*T, top_k]
    
        # get top-k in each token
        out = torch.zeros_like(x)              # [B, T, D]

        # flatten token dimension
        x_flat        = x.view(-1, D)                      # [B*T, D    ]
        topk_idx_flat = topk_idx.view(-1, self.top_k)      # [B*T, top_k]
        weights_flat  = weights.view(-1, self.top_k)       # [B*T, top_k]

        # forward per expert
        for e in range(self.num_experts):

            mask = (topk_idx_flat == e)              # [B*T, top_k] with True or False
            if not mask.any():
                continue

            token_idx, slot_idx = mask.nonzero(as_tuple=True)

            x_e = x_flat[token_idx]                  # [N, D] where N <= B*T
            w_e = weights_flat[token_idx, slot_idx]  # [N]

            y_e = self.experts[e](x_e) * w_e.unsqueeze(-1)

            out.view(-1, D)[token_idx] += y_e

        # # slow but understandable
        # gate_scores        = self.gate(x)      # [B, T, D] -> [B, T, num_experts]
        # topk_val, topk_idx = torch.topk(gate_scores, self.top_k, dim=-1)  # [B, T, top_k]
        # # build weights
        # weights = F.softmax(topk_val, dim=-1)  # [B*T, top_k]
        # for i in range(self.top_k):
        #     expert_idx    = topk_idx[..., i].unsqueeze(-1)             # [batch, seq_len, 1]
        #     expert_weight = weights [..., i].unsqueeze(-1)             # [batch, seq_len, 1]
        #     for b in range(x.size(0)):
        #         for t in range(x.size(1)):
        #             e            = int(expert_idx[b, t])
        #             out[b, t, :] = out[b, t, :] + self.experts[e](x[b, t, :]) * expert_weight[b, t]
    
        return out




class build_model(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 reward_size,
                 feature_size,
                 history_size,
                 future_size,
                 neural_type,
                 num_layers,
                 num_heads,
                 num_experts,
                 moe_top_k,
                 init,
                 opti,
                 loss,
                 bias,
                 drop_rate,
                 alpha,
                 L2_lambda,
                 grad_clip_value):

        super(build_model, self).__init__()

        self.state_size           = state_size
        self.action_size          = action_size
        self.reward_size          = reward_size
        self.feature_size         = feature_size
        self.history_size         = history_size
        self.future_size          = future_size
        self.neural_type          = neural_type
        self.num_layers           = num_layers
        self.num_heads            = num_heads
        self.num_experts          = num_experts
        self.moe_top_k            = moe_top_k
        self.init                 = init
        self.opti                 = opti
        self.loss                 = loss
        self.bias                 = bias
        self.drop_rate            = drop_rate
        self.alpha                = alpha
        self.L2_lambda            = L2_lambda
        self.grad_clip_value      = grad_clip_value

        self.state_linear         = nn.Sequential(
                                        nn.Linear(self.state_size, self.feature_size, bias=self.bias)
                                    )
        self.action_linear        = nn.Sequential(
                                        nn.Linear(self.action_size, self.feature_size, bias=self.bias)
                                    )
        self.state_norm           = nn.LayerNorm(self.feature_size, elementwise_affine=True)
        self.action_norm          = nn.LayerNorm(self.feature_size, elementwise_affine=True)

        self.state_type           = nn.Parameter(torch.randn(1, 1, self.feature_size))
        self.action_type          = nn.Parameter(torch.randn(1, 1, self.feature_size))

        self.history_type         = nn.Parameter(torch.randn(1, 1, self.feature_size))
        self.present_type         = nn.Parameter(torch.randn(1, 1, self.feature_size))
        self.future_type          = nn.Parameter(torch.randn(1, 1, self.feature_size))

        self.positional_encoding  = nn.Parameter(self.generate_positional_encoding(self.history_size + 1 + self.future_size , self.feature_size ), requires_grad=False)

        self.dropout              = nn.Dropout(self.drop_rate)
        self.transformer_layers   = \
        nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(self.feature_size, elementwise_affine=True),
                custom_attn(self.feature_size, self.num_heads, self.bias, self.drop_rate),
                nn.LayerNorm(self.feature_size, elementwise_affine=True),
                # nn.Sequential(
                #     nn.Linear(self.feature_size, self.feature_size, bias=self.bias),
                #     nn.GELU(),
                #     nn.Linear(self.feature_size, self.feature_size, bias=self.bias)
                # )
                moe_ffn(self.feature_size, num_experts=self.num_experts, top_k=self.moe_top_k, bias=self.bias)
            ])
            for _ in range(self.num_layers)
        ])
        self.transformer_norm     = nn.LayerNorm(self.feature_size, elementwise_affine=True) 
        mask                      = torch.full((1, 1, self.history_size + 1 + self.future_size, self.history_size + 1 + self.future_size), float("-inf"))
        mask                      = torch.triu(mask , diagonal=1)
        self.register_buffer('mask', mask)  

        self.reward_linear        = nn.Sequential(
                                        nn.Linear(self.feature_size, self.reward_size, bias=self.bias)
                                    )

        # Initialize weights for fully connected layers
        self.initialize_weights(self.init  )

        # Optimizer
        optimizers = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            'adamw': optim.AdamW
        }
        self.selected_optimizer = optimizers[self.opti.lower()](self.parameters(), lr=self.alpha, weight_decay=self.L2_lambda)

        # Loss function
        losses = {
            'mean_squared_error': torch.nn.MSELoss(reduction='mean'),
            'binary_crossentropy': torch.nn.BCELoss(reduction='mean'),
            'huber_loss': torch.nn.SmoothL1Loss(reduction='mean')
        }
        self.loss_function = losses[self.loss .lower()]

        # Loss function
        losses = {
            'mean_squared_error': torch.nn.MSELoss(reduction='none'),
            'binary_crossentropy': torch.nn.BCELoss(reduction='none'),
            'huber_loss': torch.nn.SmoothL1Loss(reduction='none')
        }
        self.loss_function_ = losses[self.loss .lower()]




    def forward(self, history_s, history_a, present_s, future_s, future_a):

        if history_s.size(1) > 0:
            history_s = self.state_norm (self.state_linear (history_s              ))
            present_s = self.state_norm (self.state_linear (present_s.unsqueeze(1) ))
            future_a  = self.action_norm(self.action_linear(future_a               ))
            history_s = history_s + self.history_type + self.state_type
            present_s = present_s + self.present_type + self.state_type
            future_a  = future_a  + self.future_type  + self.action_type
            h = torch.cat([history_s, present_s, future_a], dim=1)
        else:
            present_s = self.state_norm (self.state_linear (present_s.unsqueeze(1) ))
            future_a  = self.action_norm(self.action_linear(future_a               ))
            present_s = present_s + self.present_type + self.state_type
            future_a  = future_a  + self.future_type  + self.action_type
            h = torch.cat([present_s, future_a], dim=1)

        """
        Transformer decoder
        """
        long = h.size(1)
        HS = self.history_size
        FT = self.future_size
        h[:, :HS, :]          = h[:, :HS, :]          + self.positional_encoding[:, :HS, :]
        h[:, HS:HS+1, :]      = h[:, HS:HS+1, :]      + self.positional_encoding[:, :1,  :]
        h[:, HS+1:HS+1+FT, :] = h[:, HS+1:HS+1+FT, :] + self.positional_encoding[:, :FT, :]
        for layer in self.transformer_layers:
            attention_norm, attention_linear, fully_connected_norm, fully_connected_linear = layer
            h_ = attention_norm(h)
            h_ = attention_linear(h_, h_, h_, self.mask[:, :, :long, :long], kv_cache=None)[0]
            h_ = self.dropout(h_)
            h  = h + h_ # typical pre-norm style
            h_ = fully_connected_norm(h)
            h_ = fully_connected_linear(h_)
            h_ = self.dropout(h_)
            h  = h + h_ # typical pre-norm style
        h = self.transformer_norm(h) 
        """
        Transformer decoder
        """

        h = h[:, -future_a.size(1): , :]
        r = self.reward_linear(h)
        r = torch.tanh(r)  

        future_r = r
        future_s = torch.zeros((future_a.size(0), future_a.size(1), self.state_size), device=future_a.device, dtype=future_a.dtype)

        return future_r, future_s




    def _forward(self, history_s, history_a, present_s, future_s, future_a):
        return self.forward(history_s, history_a, present_s, future_s, future_a)

    


    def forward_(self, history_s, history_a, present_s, future_s, future_a):
        return self.forward(history_s, history_a, present_s, future_s, future_a)




    def generate_positional_encoding(self, sequence_size, feature_size):
        pe = torch.zeros(sequence_size,feature_size)
        for pos in range(sequence_size):
            for i in range(0,feature_size,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/feature_size)))
                if i + 1 < feature_size:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/feature_size)))
        return pe.unsqueeze(0)  # Shape: (1, sequence_size, feature_size)

    def initialize_weights(self, initializer):
        initializers = {
            'random_uniform': nn.init.uniform_,
            'random_normal': nn.init.normal_,
            'glorot_uniform': nn.init.xavier_uniform_,
            'glorot_normal': nn.init.xavier_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_, # since we are using nn.linear -> norm layer -> gelu , we don't really need kaiming for gelu
            'kaiming_normal': nn.init.kaiming_normal_
        }
        initializer = initializers[initializer.lower()]
        for module in self.modules():
            if isinstance(module, nn.Linear):
                initializer(module.weight)     # module.weight and module.bias are parameters
                if module.bias is not None:   
                    nn.init.zeros_(module.bias)