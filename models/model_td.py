
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
[O] KV Cache
[O] SwiGLU FFN
[O] MoE
[X] MOE router loss 

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
        self.W_q_s         = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_k_s         = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_v_s         = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_o_s         = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_q_a         = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_k_a         = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_v_a         = nn.Linear(feature_size, feature_size, bias=self.bias)
        self.W_o_a         = nn.Linear(feature_size, feature_size, bias=self.bias)
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
    
    def scaled_dot_product_attention(self, Q, K, V, mask):

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
    
    def scaled_dot_product_attention_(self, Q, K, V, mask): # faster official api
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

    def forward(self, Q, K, V, h_size, mask=None, kv_cache=None, seq_positions=None):
        # mask Shape: (batch_size, 1, sequence_size, sequence_size)
        # Q    Shape: (batch_size,    sequence_size, feature_size )

        Q_s  = self.W_q_s(Q[:, :h_size, :])
        K_s  = self.W_k_s(K[:, :h_size, :])
        V_s  = self.W_v_s(V[:, :h_size, :])

        Q_a  = self.W_q_a(Q[:, h_size:, :])
        K_a  = self.W_k_a(K[:, h_size:, :])
        V_a  = self.W_v_a(V[:, h_size:, :])

        Q    = torch.cat([Q_s, Q_a], dim=1)
        K    = torch.cat([K_s, K_a], dim=1)
        V    = torch.cat([V_s, V_a], dim=1)

        Q    = self.split_heads(Q)  # Shape: (batch_size, num_heads, sequence_size, head_size )
        K    = self.split_heads(K)  # Shape: (batch_size, num_heads, sequence_size, head_size )
        V    = self.split_heads(V)  # Shape: (batch_size, num_heads, sequence_size, head_size )

        # # RoPE
        # Q = self.apply_rope(Q, seq_positions)
        # K = self.apply_rope(K, seq_positions)

        if kv_cache is not None:
            if 'k' in kv_cache and 'v' in kv_cache:
                K = torch.cat([kv_cache['k'], K], dim=2)
                V = torch.cat([kv_cache['v'], V], dim=2)
            kv_cache['k'] = K
            kv_cache['v'] = V
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = self.combine_heads(attn_output)
        output_s    = self.W_o_s(attn_output[:, :h_size, :])
        output_a    = self.W_o_a(attn_output[:, h_size:, :])
        output      = torch.cat([output_s, output_a], dim=1)
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
                                        nn.Linear(self.state_size, self.feature_size, bias=self.bias),
                                        nn.GELU(),
                                        nn.Linear(self.feature_size, self.feature_size, bias=self.bias)
                                    )
        self.action_linear        = nn.Sequential(
                                        nn.Linear(self.action_size, self.feature_size, bias=self.bias),
                                        nn.GELU(),
                                        nn.Linear(self.feature_size, self.feature_size, bias=self.bias)
                                    )
        self.action_linear_       = nn.Sequential(
                                        nn.Linear(self.action_size, self.feature_size, bias=self.bias),
                                        nn.GELU(),
                                        nn.Linear(self.feature_size, self.feature_size, bias=self.bias)
                                    )
        self.history_norm         = rms_norm(self.feature_size, elementwise_affine=True)
        self.future_norm          = rms_norm(self.feature_size, elementwise_affine=True)

        self.positional_encoding  = nn.Parameter(self.generate_positional_encoding(self.history_size + self.future_size , self.feature_size ), requires_grad=False)

        self.dropout              = nn.Dropout(self.drop_rate)
        self.transformer_layers   = \
        nn.ModuleList([
            rms_norm(self.feature_size, elementwise_affine=True),
            rms_norm(self.feature_size, elementwise_affine=True),
            custom_attn(self.feature_size, self.num_heads, self.bias, self.drop_rate),
            rms_norm(self.feature_size, elementwise_affine=True),
            rms_norm(self.feature_size, elementwise_affine=True),
            nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size, bias=self.bias),
                nn.GELU(),
                nn.Linear(self.feature_size, self.feature_size, bias=self.bias)
            ),
            nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size, bias=self.bias),
                nn.GELU(),
                nn.Linear(self.feature_size, self.feature_size, bias=self.bias)
            )
            # moe_ffn(self.feature_size, num_experts=self.num_experts, top_k=self.moe_top_k, bias=self.bias)
        ])
        self.transformer_norm = rms_norm(self.feature_size, elementwise_affine=True) 
        mask           = torch.full((1, 1, self.history_size + self.future_size, self.history_size + self.future_size),float("-inf"))
        mask           = torch.triu(mask, diagonal=1)
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




    def forward(self, history_s, history_a, present_s, future_s, future_a, pos_skip):

        if history_s.size(1) > 0:
            history = self.state_linear(history_s) + self.action_linear(history_a)
            present = self.state_linear(present_s.unsqueeze(1)) + self.action_linear(future_a[:, :1, :])
            history = torch.cat([history, present], dim=1)
            history = self.history_norm(history)
        else:
            history = torch.empty((present_s.size(0), 0, self.featuer_size), device=present_s.device, dtype=present_s.dtype)
            present = self.state_linear(present_s.unsqueeze(1)) + self.action_linear(future_a[:, :1, :])
            history = torch.cat([history, present], dim=1)
            history = self.history_norm(history)
        future      = self.future_norm(self.action_linear_(future_a[:, 1:, :]))

        h = history + self.positional_encoding[:, :self.history_size + 1, :]
        f = future  + self.positional_encoding[:, self.history_size + 1:, :]

        """
        Transformer decoder
        """
        attention_norm_h, attention_norm_f, attention_linear, fully_connected_norm_h, fully_connected_norm_f, fully_connected_linear_h, fully_connected_linear_f = self.transformer_layers
        for _ in self.transformer_layers:
            h_  = attention_norm_h(h)
            f_  = attention_norm_f(f)
            hf_ = torch.cat([h_, f_], dim=1)
            hf_ = attention_linear(hf_, hf_, hf_, self.history_size + 1, mask=self.mask, kv_cache=None, seq_positions=None)[0]
            hf_ = self.dropout(hf_)
            h_  = hf_[:, :self.history_size + 1, :]
            f_  = hf_[:, self.history_size + 1:, :]
            h   = h + h_
            f   = f + f_
            h_ = fully_connected_norm_h(h)
            f_ = fully_connected_norm_f(f)
            h_ = fully_connected_linear_h(h_)
            f_ = fully_connected_linear_f(f_)
            h_ = self.dropout(h_)
            f_ = self.dropout(f_)
            h   = h + h_
            f   = f + f_
        hf = torch.cat([h, f], dim=1)
        hf = self.transformer_norm(hf) 
        """
        Transformer decoder
        """

        hf = hf[:, self.history_size:, :]
        r  = self.reward_linear(hf)
        r  = torch.tanh(r)  

        future_r = r
        future_s = torch.zeros((future_a.size(0), future_a.size(1), self.state_size), device=future_a.device, dtype=future_a.dtype)

        return future_r, future_s




    def _forward(self, history_s, history_a, present_s, future_s, future_a, pos_skip):
        return self.forward(history_s, history_a, present_s, future_s, future_a, pos_skip)

    


    def forward_(self, history_s, history_a, present_s, future_s, future_a):
        return self.forward(history_s, history_a, present_s, future_s, future_a, None)




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