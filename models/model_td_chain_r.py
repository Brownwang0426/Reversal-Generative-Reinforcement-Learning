
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

--- check list for modern transformer's basic architecture, [O] means included in this repo, [X] means not yet included ---
[O] Pre-RMSNorm
[X] RoPE
[O] causal mask
[O] GQA or MHA
[X] Flash / SDPA Attention
[X] KV Cache
[X] SwiGLU FFN
[X] MoE
[X] MoE router loss

--- optional ---
[X] Residual scaling
[X] AdamW
[X] Weight tying

"""

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
        return x.view(batch_size, sequence_size, self.num_heads, self.head_size).contiguous().transpose(1, 2).contiguous()

    def apply_rope(self, x, seq_positions):
        """
        x            : [batch, num_heads, seq_len, head_size]
        seq_positions: [seq_len] or [batch, seq_len]
        """
        B, H, L, D = x.shape
        device = x.device

        d_half = D // 2
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # --- auto generate seq_positions if None ---
        if seq_positions is None:
            seq_positions = torch.arange(L, device=device)  # [L]

        freqs = 1.0 / (10000 ** (torch.arange(0, d_half, device=device) / d_half))
        if seq_positions.dim() == 1:
            theta = seq_positions[:, None] * freqs[None, :]  # [L, D/2]
        else:
            theta = seq_positions[:, :, None] * freqs[None, None, :]  # [B, L, D/2]

        # reshape broadcast
        theta = theta  # [B, 1, L, D/2] or [1, 1, L, D/2]

        # rotation
        x_rot = torch.zeros_like(x)
        x_rot[..., 0::2] = x1 * torch.cos(theta) - x2 * torch.sin(theta)
        x_rot[..., 1::2] = x1 * torch.sin(theta) + x2 * torch.cos(theta)
        return x_rot

    def scaled_dot_product_attention_(self, Q, K, V, mask):

        # attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5) #  (batch_size, num_heads, sequence_size, head_size) @ (batch_size, num_heads, head_size, sequence_size )
        K_T = K.transpose(-2, -1).contiguous()
        attn_scores = (Q @ K_T) / (self.head_size ** 0.5)

        if mask is not None:
            attn_scores = attn_scores + mask                   # (batch_size, num_heads, sequence_size, sequence_size) += (batch_size, 1, sequence_size, sequence_size)
        else:
            pass

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout (attn_probs)
        # output     = torch.matmul(attn_probs, V)  # (batch_size, num_heads, sequence_size, sequence_size) @ (batch_size, num_heads, sequence_size, head_size )
        output     = attn_probs @ V
        return output                               # (batch_size, num_heads, sequence_size, head_size)

    def scaled_dot_product_attention(self, Q, K, V, mask): # faster official api
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        ):
            return F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,
                dropout_p=self.drop_rate,
                is_causal=True
            )

    def combine_heads(self, x):
        batch_size, num_heads, sequence_size, head_size = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_size, self.feature_size).contiguous()

    def forward(self, Q, K, V, mask=None, kv_cache=None, seq_positions=None):
        # mask Shape: (batch_size, 1, sequence_size, sequence_size)
        # Q    Shape: (batch_size,    sequence_size, feature_size )
        Q    = self.split_heads(self.W_q(Q))  # Shape: (batch_size, num_heads, sequence_size, head_size )
        K    = self.split_heads(self.W_k(K))  # Shape: (batch_size, num_heads, sequence_size, head_size )
        V    = self.split_heads(self.W_v(V))  # Shape: (batch_size, num_heads, sequence_size, head_size )

        # # RoPE
        # Q = self.apply_rope(Q, seq_positions)
        # K = self.apply_rope(K, seq_positions)

        if kv_cache is not None:
            if 'k' in kv_cache and 'v' in kv_cache:
                K = torch.cat([kv_cache['k'], K], dim=2)
                V = torch.cat([kv_cache['v'], V], dim=2)
            kv_cache['k'] = K
            kv_cache['v'] = V
        attn_output = self.scaled_dot_product_attention_(Q, K, V, mask)
        output      = self.W_o(self.combine_heads(attn_output))
        return output, kv_cache




class rms_norm(nn.Module):
    def __init__(self, dim, elementwise_affine=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        if self.weight is not None:
            x = x * self.weight
        return x




class swiss_glu_ffn(nn.Module):
    def __init__(self, feature_size, bias=True):
        super().__init__()
        self.w_gate  = nn.Linear(feature_size, feature_size, bias=bias)
        self.w_value = nn.Linear(feature_size, feature_size, bias=bias)
        self.w_out   = nn.Linear(feature_size, feature_size, bias=bias)

    def forward(self, x):
        return self.w_out(
            F.silu(self.w_gate(x)) * self.w_value(x)
        )

class moe_ffn(nn.Module):
    def __init__(self, feature_size, num_experts=4, top_k=2, bias=False):
        super(moe_ffn, self).__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.bias        = bias
        self.experts     = nn.ModuleList([
            swiss_glu_ffn(feature_size, bias=self.bias)
            for _ in range(num_experts)
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
        x_flat        = x.view(-1, D).contiguous()                      # [B*T, D    ]
        topk_idx_flat = topk_idx.view(-1, self.top_k).contiguous()      # [B*T, top_k]
        weights_flat  = weights.view(-1, self.top_k).contiguous()       # [B*T, top_k]

        # forward per expert
        for e in range(self.num_experts):

            mask = (topk_idx_flat == e)              # [B*T, top_k] with True or False
            if not mask.any():
                continue

            token_idx, slot_idx = mask.nonzero(as_tuple=True)

            x_e = x_flat[token_idx]                  # [N, D] where N <= B*T
            w_e = weights_flat[token_idx, slot_idx]  # [N]

            y_e = self.experts[e](x_e) * w_e.unsqueeze(-1)

            out.view(-1, D).contiguous()[token_idx] += y_e

        return out




class build_model(nn.Module):
    def __init__(self,
                 reward_size,
                 state_size,
                 action_size,
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

        self.reward_size          = reward_size
        self.state_size           = state_size
        self.action_size          = action_size
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

        self.reward_linear        = nn.Sequential(
                                        nn.Linear(self.reward_size, self.feature_size, bias=self.bias)
                                    )
        self.state_linear         = nn.Sequential(
                                        nn.Linear(self.state_size, self.feature_size, bias=self.bias)
                                    )
        self.action_linear        = nn.Sequential(
                                        nn.Linear(self.action_size, self.feature_size, bias=self.bias)
                                    )
        self.reward_norm          = rms_norm(self.feature_size, elementwise_affine=True)
        self.state_norm           = rms_norm(self.feature_size, elementwise_affine=True)
        self.action_norm          = rms_norm(self.feature_size, elementwise_affine=True)

        self.positional_encoding  = nn.Parameter(self.generate_positional_encoding((self.history_size + 1 + self.future_size) * 1, self.feature_size ), requires_grad=False)

        self.dropout              = nn.Dropout(self.drop_rate)
        self.transformer_layers   = \
        nn.ModuleList([
            nn.ModuleList([
                rms_norm(self.feature_size, elementwise_affine=True),
                custom_attn(self.feature_size, self.num_heads, self.bias, self.drop_rate),
                rms_norm(self.feature_size, elementwise_affine=True),
                nn.Sequential(
                    nn.Linear(self.feature_size, self.feature_size, bias=self.bias),
                    nn.GELU(),
                    nn.Linear(self.feature_size, self.feature_size, bias=self.bias)
                )
                # moe_ffn(self.feature_size, num_experts=self.num_experts, top_k=self.moe_top_k, bias=self.bias)
            ])
            for _ in range(self.num_layers)
        ])
        self.transformer_norm     = rms_norm(self.feature_size, elementwise_affine=True)
        seq_len                   = (self.history_size + 1 + self.future_size) * 1
        group_idx                 = torch.arange(seq_len) // 1
        mask                      = torch.where(group_idx.unsqueeze(0) > group_idx.unsqueeze(1), float("-inf"), 0.0)
        mask                      = mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer('mask', mask)

        self.reward_linear_       = nn.Sequential(
                                        nn.Linear(self.feature_size, self.reward_size, bias=self.bias)
                                    )
        self.state_linear_        = nn.Sequential(
                                        nn.Linear(self.feature_size, self.state_size, bias=self.bias)
                                    )
        self.reward_norm_         = rms_norm(self.reward_size , elementwise_affine=True)
        self.state_norm_          = rms_norm(self.state_size  , elementwise_affine=True)

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




    def forward(self, history_r, history_s, history_a, present_r, present_s, present_a, future_r, future_s, future_a):

        future_r_list = list()
        future_s_list = list()

        window_list   = list()
        if history_s.size(1) > 0:
            history_r = self.reward_norm (self.reward_linear (history_r) )
            history_s = self.state_norm  (self.state_linear  (history_s) )
            history_a = self.action_norm (self.action_linear (history_a) )
            for i in range(history_s.size(1)):
                window_list.append(history_r[:, i:i+1] + history_s[:, i:i+1] + history_a[:, i:i+1])

        present_r = self.reward_norm(self.reward_linear (present_r))
        present_s = self.state_norm (self.state_linear  (present_s))
        present_a = self.action_norm(self.action_linear (present_a))

        for i in range(future_a.size(1)):

            window_list.append(present_r + present_s + present_a)
            h = torch.cat(window_list, dim=1)
            
            """
            Transformer decoder with pre-norm style without gelu
            """
            long = h.size(1)
            h    = h + self.positional_encoding[:, :long, :]
            for layer in self.transformer_layers:
                attention_norm, attention_linear, fully_connected_norm, fully_connected_linear = layer
                h_  = attention_norm(h) 
                h_  = attention_linear(h_, h_, h_, mask=self.mask[:, :, :long, :long], kv_cache=None, seq_positions=None)[0]
                h_  = self.dropout(h_)
                h   = h + h_ # typical pre-norm style
                h_  = fully_connected_norm(h)
                h_  = fully_connected_linear(h_)
                h_  = self.dropout(h_)
                h   = h + h_ # typical pre-norm style
            h  = self.transformer_norm(h)
            """
            We utilize the last idx in h to derive the latest reward and state.
            """

            h = h[:, -1:, :]
            r = self.reward_linear_(h)
            r = self.reward_norm_(r)
            s = self.state_linear_(h)
            s = self.state_norm_(s)

            future_r_list.append(r)
            future_s_list.append(s)

            present_r = self.reward_norm(self.reward_linear(r))
            present_s = self.state_norm (self.state_linear (s))
            present_a = self.action_norm(self.action_linear(future_a[:, i:i+1]))  

        future_r = torch.cat(future_r_list, dim=1) # future_r becomes [batch_size, sequence_size, reward_size]
        future_s = torch.cat(future_s_list, dim=1) # future_s becomes [batch_size, sequence_size, state_size ]
    
        return future_r, future_s




    # def _forward(self, history_s, history_a, present_s, future_s, future_a):
    # 
    #     future_r_list = list()
    #     future_s_list = list()
    # 
    #     present_s = present_s.unsqueeze(1)
    # 
    # 
    #     if history_s.size(1) > 0:
    #         history_s   = self.state_norm (self.state_linear (history_s) )
    #         history_a   = self.action_norm(self.action_linear(history_a) )
    #         history_s_a = history_s + history_a 
    #     else:
    #         history_s_a = torch.empty((present_s.size(0), 0, self.feature_size), device=present_s.device, dtype=present_s.dtype)
    #             
    # 
    #     present_s = self.state_norm (self.state_linear (present_s))
    #     future_a  = self.action_norm(self.action_linear(future_a ))
    # 
    # 
    #     kv_caches = [dict() for _ in self.transformer_layers]
    #     start     = 0
    # 
    #     for i in range(int(future_a.size(1))):
    # 
    #         h = torch.cat([history_s_a, present_s + future_a[:, i:i+1]], dim=1)
    # 
    #         """
    #         Transformer decoder
    #         """
    #         end  = start + history_s_a.size(1) + 1
    #         h    = h + self.positional_encoding[:, start : end , :] # [todo]
    #         for j, layer in enumerate(self.transformer_layers):
    #             attention_norm, attention_linear, fully_connected_norm, fully_connected_linear = layer
    #             h_  = attention_norm(h)
    #             seq_positions = torch.arange(start, end, device=h.device) # [seq_len]
    #             h_, kv_caches[j] = attention_linear(h_, h_, h_, mask=self.mask[:, :, start : end, : end], kv_cache=kv_caches[j], seq_positions=seq_positions)
    #             h_  = self.dropout(h_)
    #             h   = h + h_
    #             h_  = fully_connected_norm(h)
    #             h_  = fully_connected_linear(h_)
    #             h_  = self.dropout(h_)
    #             h   = h + h_
    #         h = self.transformer_norm(h)
    #         """
    #         Transformer decoder
    #         """
    # 
    #         h = h[:, -1:, :]
    #         r = self.reward_linear(h)
    #         r = (r)  
    #         s = self.state_linear_(h)
    #         s = (s)  
    # 
    #         future_r_list.append(r)
    #         future_s_list.append(s)
    # 
    #         present_s = self.state_norm(self.state_linear(s)) 
    # 
    #         history_s_a = torch.empty((present_s.size(0), 0, self.feature_size), device=present_s.device, dtype=present_s.dtype)
    #         start       = copy.deepcopy(end) 
    #         
    #     future_r = torch.cat(future_r_list, dim=1) 
    #     future_s = torch.cat(future_s_list, dim=1)
    # 
    #     return future_r, future_s

    


    def forward_(self, history_r, history_s, history_a, present_r, present_s, present_a, future_r, future_s, future_a):

        if history_s.size(1) > 0:
            history_r   = self.reward_norm(self.reward_linear(history_r) )
            history_s   = self.state_norm (self.state_linear (history_s) )
            history_a   = self.action_norm(self.action_linear(history_a) )
            history     = history_r + history_s + history_a
        else:
            history     = torch.empty((present_s.size(0), 0, self.feature_size), device=present_s.device, dtype=present_s.dtype)

        present_r   = self.reward_norm(self.reward_linear(present_r) )
        present_s   = self.state_norm (self.state_linear (present_s) )
        present_a   = self.action_norm(self.action_linear(present_a) )
        present     = present_r + present_s + present_a

        future_r    = self.reward_norm(self.reward_linear(future_r[:, :-1, :]) )
        future_s    = self.state_norm (self.state_linear (future_s[:, :-1, :]) )
        future_a    = self.action_norm(self.action_linear(future_a[:, :-1, :]) )
        future      = future_r + future_s + future_a

        h = torch.cat([history, present, future], dim=1)

        """
        Transformer decoder
        """
        long = h.size(1)
        h = h + self.positional_encoding[:, :long, :]
        for layer in self.transformer_layers:
            attention_norm, attention_linear, fully_connected_norm, fully_connected_linear = layer
            h_  = attention_norm(h)
            h_  = attention_linear(h_, h_, h_, mask=self.mask[:, :, :long, :long], kv_cache=None, seq_positions=None)[0]
            h_  = self.dropout(h_)
            h   = h + h_
            h_  = fully_connected_norm(h)
            h_  = fully_connected_linear(h_)
            h_  = self.dropout(h_)
            h   = h + h_
        h = self.transformer_norm(h)
        """
        Transformer decoder
        """

        h = h[:, -1-future_r.size(1):, :]
        r = self.reward_linear_(h)
        r = self.reward_norm_(r)
        s = self.state_linear_ (h)
        s = self.state_norm_(s)

        future_r = r
        future_s = s

        return future_r, future_s




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
            'kaiming_uniform': nn.init.kaiming_uniform_, 
            'kaiming_normal': nn.init.kaiming_normal_
        }
        initializer = initializers[initializer.lower()]
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                initializer(module.weight)     # module.weight and module.bias are parameters
                if module.bias is not None:   
                    nn.init.zeros_(module.bias)
                # if "reward_linear" in name:
                #     if module.bias is not None:
                #         nn.init.constant_(module.bias, 2.0)  # ★ key to make agent optimisitc and explore
                # else:
                #     initializer(module.weight)     # module.weight and module.bias are parameters
                #     if module.bias is not None:   
                #         nn.init.zeros_(module.bias)