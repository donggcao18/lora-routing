# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.llama.configuration_llama import LlamaConfig

import torch.distributed as dist
import torch.multiprocessing as mp

from cl_dataset import GaussianDistribution

from assets import merge_distributions
import torch.nn.functional as F


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.
    ):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        self.out_features = out_features

        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        self.scaling = self.lora_alpha / self.r
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor):
        result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result.reshape(x.shape[0], -1, self.out_features)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, prompt_config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        # self.successor=False

        self.distances_way=prompt_config['distances_way']
        self.distances_temperature=prompt_config['distances_temperature']

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.lora_q = LoRALayer(self.hidden_size, self.num_heads * self.head_dim, r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"], lora_dropout=prompt_config["lora_dropout"])
        self.distribution_q=GaussianDistribution()
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.lora_v = LoRALayer(self.hidden_size, self.num_heads * self.head_dim, r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"], lora_dropout=prompt_config["lora_dropout"])
        self.distribution_v=GaussianDistribution()
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        self.previous_lora_weights_q, self.previous_lora_weights_v = None, None
        self.previous_lora_distribution_q= None
        self.previous_lora_distribution_v= None

        self.key_attention_weights_q=None
        self.key_attention_weights_v=None

        self.log_key_attention_weights_q=None
        self.log_key_attention_weights_v=None

        self.train_key_weight_top=prompt_config["train_key_weight_top"]
        self.test_key_weight_top=prompt_config["test_key_weight_top"]

        self.train_key_weight_top_p=prompt_config["train_key_weight_top_p"]
        self.test_key_weight_top_p=prompt_config["test_key_weight_top_p"]

        # self.flag=True

        self.prompt_config = prompt_config
        if prompt_config["previous_lora_path"] is not None:
            previous_lora_list = prompt_config["previous_lora_path"].split(',')

            with torch.no_grad():
                self.previous_lora_weights_q = nn.ModuleList()
                for i in range(len(previous_lora_list)):
                    layer = LoRALayer(self.hidden_size, self.num_heads * self.head_dim, r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"], lora_dropout=prompt_config["lora_dropout"])
                    self.previous_lora_weights_q.append(layer)

                self.previous_lora_weights_v = nn.ModuleList()
                for i in range(len(previous_lora_list)):
                    layer = LoRALayer(self.hidden_size, self.num_heads * self.head_dim, r=prompt_config["lora_r"], lora_alpha=prompt_config["lora_alpha"], lora_dropout=prompt_config["lora_dropout"])
                    self.previous_lora_weights_v.append(layer)
        
        if prompt_config["previous_lora_distribution_path"] is not None:
            with torch.no_grad():
                previous_lora_list = prompt_config["previous_lora_distribution_path"].split(',')
                self.previous_lora_distribution_q=[]
                self.previous_lora_distribution_v=[]
                for i in range(len(previous_lora_list)):
                    self.previous_lora_distribution_q.append(GaussianDistribution())
                    self.previous_lora_distribution_v.append(GaussianDistribution())

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def agg_lora_states(self, hidden_states, lora_layer, pre_lora_layer, key_attention_weights):
        bsz, q_len, _ = hidden_states.size()
        _, num_task, _ = key_attention_weights.size()
        if pre_lora_layer is not None and num_task > 1:
            cur_lora_states = lora_layer(hidden_states).unsqueeze(0)
            with torch.no_grad():
                pre_lora_states = torch.cat([pre_lora(hidden_states).unsqueeze(0) for pre_lora in pre_lora_layer], dim=0)
            concat_q = torch.cat([cur_lora_states, pre_lora_states], dim=0).transpose(0, 1).reshape(bsz, -1, hidden_states.shape[1]*self.num_heads * self.head_dim)

            agg_lora_states = torch.matmul(key_attention_weights.transpose(1, 2), concat_q).squeeze()

        else:
            cur_lora_states = lora_layer(hidden_states).unsqueeze(0).transpose(0, 1).reshape(bsz, -1, hidden_states.shape[1]*self.num_heads * self.head_dim)
            agg_lora_states = torch.matmul(key_attention_weights.transpose(1, 2), cur_lora_states).squeeze()

        return agg_lora_states.reshape(bsz, -1, self.num_heads * self.head_dim)


    def calculate_distances(self, features, distributions, distance_type='L2', temperature=1.0):
        with torch.no_grad():
            distances = []
            for gaussian in distributions:
                mean = gaussian.mean
                var = gaussian.var
                if distance_type == 'L2':
                    dist = torch.norm(features - mean, dim=1)
                elif distance_type == 'Gaussian':
                    var = var + 1e-6 
                    log_prob = -0.5 * (torch.log(2 * torch.pi * var) + ((features - mean) ** 2) / var)
                    log_prob_sum = log_prob.sum(dim=1)  # [B]
                    dist = log_prob_sum
                elif distance_type == 'Attention':
                    dim = features.size(-1)
                    ty=features.dtype
                    attn_score = torch.matmul(features, mean.type(ty)) / torch.sqrt(torch.tensor(dim))
                    dist = attn_score  # [B]
                elif distance_type == 'Cosine':
                    features_norm = F.normalize(features, dim=1)
                    mean_norm = F.normalize(mean.reshape(1, -1), dim=1)
                    dist = torch.matmul(features_norm, mean_norm.t())  # [B, 1]
                    dist=dist.reshape(-1)
                else:
                    raise ValueError(f"Unsupported distance type: {distance_type}")
                distances.append(dist)
            
            if distance_type == 'Gaussian':
                dis_weights = torch.stack(distances, dim=1)  # [B, num_distributions]
                dis_weights = F.softmax(dis_weights / temperature, dim=1)
            elif distance_type == 'Attention':
                dis_weights = torch.stack(distances, dim=1)  # [B, num_distributions]
                dis_weights = F.softmax(dis_weights / temperature, dim=1)
            elif distance_type == 'L2':
                dis_weights = torch.stack(distances, dim=1)  # [B, num_distributions]
                dis_weights = F.softmax(-dis_weights / temperature, dim=1)
            elif distance_type == 'Cosine':
                dis_weights = torch.stack(distances, dim=1)
                dis_weights = F.softmax(dis_weights / temperature, dim=1)
            else:
                raise ValueError(f"Unsupported distance type: {distance_type}")
            
            return dis_weights.unsqueeze(-1)  # [B, num_distributions, 1]
        
    def top_k_weights(self,key_weights,top_k):
        if top_k>key_weights.shape[1]:
            top_k=key_weights.shape[1]
        topk_values, topk_indices = torch.topk(key_weights.squeeze(-1), top_k, dim=1)

        topk_weights = torch.zeros_like(key_weights)
        topk_weights.scatter_(1, topk_indices.unsqueeze(-1), topk_values.unsqueeze(-1))
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
        return topk_weights


    def top_p_weights(self,key_weights, top_p, norm=True):
        """
        Performs Top-p (nucleus) filtering on the input key_weights.

        Args:
            key_weights (torch.Tensor): Tensor of shape (B, K, 1) containing softmax probabilities.
            top_p (float): Cumulative probability threshold (0 < top_p <= 1).

        Returns:
            torch.Tensor: Tensor of shape (B, K, 1) after applying Top-p filtering and renormalization.
        """
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in the range (0.0, 1.0].")

        B, K, _ = key_weights.shape

        # Remove the last dimension for processing
        key_weights = key_weights.squeeze(-1)  # Shape: (B, K)

        # Sort the probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(key_weights, descending=True, dim=1)  # Both (B, K)

        # Compute the cumulative sum of the sorted probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=1)  # Shape: (B, K)

        # Identify where cumulative_probs exceeds top_p
        exceeds_p = cumulative_probs >= top_p  # Shape: (B, K), bool

        # For each batch, find the first index where cumulative_probs > top_p
        # If top_p is 1.0, ensure all tokens are included
        if top_p < 1.0:
            # Replace the first occurrence of True to mark the cutoff
            cutoff_indices = exceeds_p.float().argmax(dim=1)  # Shape: (B,)
            # If top_p is very large and not exceeded, argmax will return 0, so we set it to K
            # where cumulative_probs[-1] > top_p is always True if top_p <1.0
            cutoff_indices = torch.where(exceeds_p.any(dim=1), cutoff_indices, torch.full_like(cutoff_indices, K-1))
        else:
            # If top_p ==1.0, include all tokens
            cutoff_indices = torch.full((B,), K-1, dtype=torch.long, device=key_weights.device)

        # Create a mask for each batch
        range_tensor = torch.arange(K, device=key_weights.device).unsqueeze(0).expand(B, K)  # Shape: (B, K)
        cutoff_indices = cutoff_indices.unsqueeze(1)  # Shape: (B, 1)
        mask = range_tensor <= cutoff_indices  # Shape: (B, K)

        # Apply the mask to sorted_probs
        filtered_sorted_probs = sorted_probs * mask.float()  # Zero out probabilities beyond top_p

        # Re-normalize the filtered probabilities
        sum_probs = filtered_sorted_probs.sum(dim=1, keepdim=True)  # Shape: (B, 1)
        # To avoid division by zero, clamp the sum to a minimum value
        sum_probs = sum_probs.clamp(min=1e-10)
        if norm==True:
            normalized_sorted_probs = filtered_sorted_probs / sum_probs  # Shape: (B, K)
        else:
            normalized_sorted_probs = filtered_sorted_probs
        normalized_sorted_probs=normalized_sorted_probs.to(key_weights.dtype)

        # Initialize a tensor of zeros for the final top_p weights
        top_p_weights = torch.zeros_like(key_weights)  # Shape: (B, K)

        # Scatter the normalized sorted probabilities back to their original indices

        # print(f"sorted_indices: {sorted_indices}")
        # print(f"top_p_weights: {top_p_weights}")
        # print(f"normalized_sorted_probs: {normalized_sorted_probs}")
        # print(f"sorted_indices.dtype: {sorted_indices.dtype}")
        # print(f"normalized_sorted_probs.dtype: {normalized_sorted_probs.dtype}")
        # print(f"top_p_weights.dtype: {top_p_weights.dtype}")
        
        top_p_weights.scatter_(1, sorted_indices, normalized_sorted_probs)

        # Restore the last dimension
        top_p_weights = top_p_weights.unsqueeze(-1)  # Shape: (B, K, 1)

        return top_p_weights
    
    def updata_distribution_q(self,all_gpu_hidden_states,all_gpu_input_ids_wo_label,all_gpu_input_ids):
        self.up_q_list=[]
        if (self.distribution_q is not None) and self.training:
            with torch.no_grad():
                for each_q,each_ids_w,each_ids in zip(all_gpu_hidden_states,all_gpu_input_ids_wo_label,all_gpu_input_ids):
                    each_q=self.q_proj(each_q.unsqueeze(0)).squeeze(0)
                    each_q=each_q[(each_ids==1).long().sum():len(each_ids_w)-(each_ids_w==1).long().sum()+(each_ids==1).long().sum()]
                    each_q=torch.mean(each_q,dim=0)
                    if torch.isnan(each_q).any():
                        torch.set_printoptions(edgeitems=1000)
                        print(each_q)
                        print(each_ids_w)
                        print(each_ids)
                        raise ValueError(
                            f"/0 error"
                        )
                    self.distribution_q.update(each_q)
                    self.up_q_list.append(each_q)
        if (self.distribution_q is not None) and self.training:
            with torch.no_grad():
                if torch.cuda.device_count()>1:
                    local_rank=torch.distributed.get_rank()

                    each_gpu_up_q=self.up_q_list
                    all_gpu_up_q_list=[None] *torch.cuda.device_count()
                    torch.distributed.all_gather_object(all_gpu_up_q_list,each_gpu_up_q)
                    all_gpu_up_q=[]
                    for t in range(len(all_gpu_up_q_list)): 
                        if t !=local_rank:
                            for row in all_gpu_up_q_list[t]:
                                all_gpu_up_q.append(row.to(f'cuda:{local_rank}'))
                    for each_q in all_gpu_up_q:
                        self.distribution_q.update(each_q)

    def updata_distribution_v(self,all_gpu_hidden_states,all_gpu_input_ids_wo_label,all_gpu_input_ids):
        self.up_v_list=[]
        if (self.distribution_v is not None) and self.training:
            with torch.no_grad():
                for each_v,each_ids_w,each_ids in zip(all_gpu_hidden_states,all_gpu_input_ids_wo_label,all_gpu_input_ids):
                    each_v=self.v_proj(each_v.unsqueeze(0)).squeeze(0)
                    each_v=each_v[(each_ids==1).long().sum():len(each_ids_w)-(each_ids_w==1).long().sum()+(each_ids==1).long().sum()]
                    each_v=torch.mean(each_v,dim=0)
                    if torch.isnan(each_v).any():
                        print(each_v)
                        print(each_ids_w)
                        print(each_ids)
                        raise ValueError(
                            f"/0 error"
                        )
                    self.distribution_v.update(each_v)
                    self.up_v_list.append(each_v)
        if (self.distribution_v is not None) and self.training:
            with torch.no_grad():
                if torch.cuda.device_count()>1:
                    local_rank=torch.distributed.get_rank()

                    each_gpu_up_v=self.up_v_list

                    all_gpu_up_v_list=[None] *torch.cuda.device_count()
                    torch.distributed.all_gather_object(all_gpu_up_v_list,each_gpu_up_v)
                    all_gpu_up_v=[]
                    for t in range(len(all_gpu_up_v_list)): 
                        if t !=local_rank:
                            for row in all_gpu_up_v_list[t]:
                                all_gpu_up_v.append(row.to(f'cuda:{local_rank}'))
                    for each_v in all_gpu_up_v:
                        self.distribution_v.update(each_v)


    def calculate_key_attention_weights_q(self,hidden_states,input_ids_wo_label,input_ids):
        if (self.previous_lora_weights_q is not None) and (hidden_states.shape[1]>1):
            with torch.no_grad():
                key_q=None
                for each_q,each_ids_w,each_ids in zip(self.q_proj(hidden_states),input_ids_wo_label,input_ids):
                    each_q=each_q[(each_ids==1).long().sum():len(each_ids_w)-(each_ids_w==1).long().sum()+(each_ids==1).long().sum()]
                    each_q=torch.mean(each_q,dim=0)
                    if key_q is None:
                        key_q=each_q.unsqueeze(0)
                    else:
                        key_q = torch.cat((key_q, each_q.unsqueeze(0)), dim=0)
                self.key_attention_weights_q=self.calculate_distances(key_q,[self.distribution_q]+self.previous_lora_distribution_q,self.distances_way,self.distances_temperature)

                if (self.training) and (self.train_key_weight_top > 0):
                    self.key_attention_weights_q=self.top_k_weights(self.key_attention_weights_q,self.train_key_weight_top)
                elif (not self.training) and (self.test_key_weight_top > 0):
                    self.key_attention_weights_q=self.top_k_weights(self.key_attention_weights_q,self.test_key_weight_top)

                if (self.training) and (self.train_key_weight_top_p > 0):
                    self.key_attention_weights_q=self.top_p_weights(self.key_attention_weights_q,self.train_key_weight_top_p)
                elif (not self.training) and (self.test_key_weight_top_p > 0):
                    self.key_attention_weights_q=self.top_p_weights(self.key_attention_weights_q,self.test_key_weight_top_p)
                
                if self.log_key_attention_weights_q is not None:
                    self.log_key_attention_weights_q.append(self.key_attention_weights_q.to('cpu'))

    def calculate_key_attention_weights_v(self,hidden_states,input_ids_wo_label,input_ids):
        if (self.previous_lora_weights_v is not None) and (hidden_states.shape[1]>1):
            with torch.no_grad():
                key_v=None
                for each_v,each_ids_w,each_ids in zip(self.v_proj(hidden_states),input_ids_wo_label,input_ids):
                    each_v=each_v[(each_ids==1).long().sum():len(each_ids_w)-(each_ids_w==1).long().sum()+(each_ids==1).long().sum()]
                    each_v=torch.mean(each_v,dim=0)
                    if key_v is None:
                        key_v=each_v.unsqueeze(0)
                    else:
                        key_v = torch.cat((key_v, each_v.unsqueeze(0)), dim=0)
                self.key_attention_weights_v=self.calculate_distances(key_v,[self.distribution_v]+self.previous_lora_distribution_v,self.distances_way,self.distances_temperature)

                if (self.training) and (self.train_key_weight_top > 0):
                    self.key_attention_weights_v=self.top_k_weights(self.key_attention_weights_v,self.train_key_weight_top)
                elif (not self.training) and (self.test_key_weight_top > 0):
                    self.key_attention_weights_v=self.top_k_weights(self.key_attention_weights_v,self.test_key_weight_top)

                if (self.training) and (self.train_key_weight_top_p > 0):
                    self.key_attention_weights_v=self.top_p_weights(self.key_attention_weights_v,self.train_key_weight_top_p)
                elif (not self.training) and (self.test_key_weight_top_p > 0):
                    self.key_attention_weights_v=self.top_p_weights(self.key_attention_weights_v,self.test_key_weight_top_p)
                
                if self.log_key_attention_weights_v is not None:
                    self.log_key_attention_weights_v.append(self.key_attention_weights_v.to('cpu'))

        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        key_attention_weights: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids_wo_label: Optional[torch.LongTensor] = None,
        attention_mask_flash: Optional[torch.Tensor] = None,
        past_key_attention_weights_q: Optional[torch.Tensor] = None,
        past_key_attention_weights_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        self.updata_distribution_q(hidden_states,input_ids_wo_label,input_ids)

        if past_key_attention_weights_q is not None:
            self.key_attention_weights_q=past_key_attention_weights_q
            if self.log_key_attention_weights_q is not None:
                self.log_key_attention_weights_q.append(past_key_attention_weights_q)
        else:
            self.calculate_key_attention_weights_q(hidden_states,input_ids_wo_label,input_ids)


        if self.key_attention_weights_q is not None:
            query_states = (self.q_proj(hidden_states)+self.agg_lora_states(hidden_states, self.lora_q, self.previous_lora_weights_q, self.key_attention_weights_q)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            query_states = (self.q_proj(hidden_states)+self.lora_q(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        self.updata_distribution_v(hidden_states,input_ids_wo_label,input_ids)
        if past_key_attention_weights_v is not None:
            self.key_attention_weights_v=past_key_attention_weights_v
            if self.log_key_attention_weights_v is not None:
                self.log_key_attention_weights_v.append(past_key_attention_weights_v)
        else:
            self.calculate_key_attention_weights_v(hidden_states,input_ids_wo_label,input_ids)

        if self.key_attention_weights_v is not None:
            value_states = (self.v_proj(hidden_states)+self.agg_lora_states(hidden_states, self.lora_v, self.previous_lora_weights_v, self.key_attention_weights_v)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            value_states = (self.v_proj(hidden_states)+self.lora_v(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, config: LlamaConfig, prompt_config):
        super().__init__(config, prompt_config)
        

        self.is_causal = True
        self._flash_attn_uses_top_left_mask = False

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # print(f"Rank {torch.distributed.get_rank()}: query_length: {query_length}")
        # print(f"Rank {torch.distributed.get_rank()}: kv_seq_len: {kv_seq_len}")
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
        
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence

        # print(f"Rank {torch.distributed.get_rank()}: attention_mask: {attention_mask}")

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # print(f"Rank {torch.distributed.get_rank()}: cu_seqlens_q: {cu_seqlens_q}")
            # print(f"Rank {torch.distributed.get_rank()}: cu_seqlens_q.shape: {cu_seqlens_q.shape}")

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output



    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        key_attention_weights: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids_wo_label: Optional[torch.LongTensor] = None,
        attention_mask_flash: Optional[torch.Tensor] = None,
        past_key_attention_weights_q: Optional[torch.Tensor] = None,
        past_key_attention_weights_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        self.updata_distribution_q(hidden_states,input_ids_wo_label,input_ids)

        if past_key_attention_weights_q is not None:
            self.key_attention_weights_q=past_key_attention_weights_q
            if self.log_key_attention_weights_q is not None:
                self.log_key_attention_weights_q.append(past_key_attention_weights_q)
        else:
            self.calculate_key_attention_weights_q(hidden_states,input_ids_wo_label,input_ids)


        if self.key_attention_weights_q is not None:
            query_states = self.q_proj(hidden_states)+self.agg_lora_states(hidden_states, self.lora_q, self.previous_lora_weights_q, self.key_attention_weights_q)
        else:
            query_states = self.q_proj(hidden_states)+self.lora_q(hidden_states)
        

        key_states = self.k_proj(hidden_states)

        self.updata_distribution_v(hidden_states,input_ids_wo_label,input_ids)

        if past_key_attention_weights_v is not None:
            self.key_attention_weights_v=past_key_attention_weights_v
            if self.log_key_attention_weights_v is not None:
                self.log_key_attention_weights_v.append(past_key_attention_weights_v)
        else:
            self.calculate_key_attention_weights_v(hidden_states,input_ids_wo_label,input_ids)

        if self.key_attention_weights_v is not None:
            value_states = self.v_proj(hidden_states)+self.agg_lora_states(hidden_states, self.lora_v, self.previous_lora_weights_v, self.key_attention_weights_v)
        else:
            value_states = self.v_proj(hidden_states)+self.lora_v(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        dropout_rate = 0.0


        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)


        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask_flash, q_len, dropout=dropout_rate
        )


        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value





class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, prompt_config):
        super().__init__()
        self.hidden_size = config.hidden_size
        if prompt_config['flash_attention']==True:
            self.self_attn = LlamaFlashAttention2(config=config, prompt_config=prompt_config)
        else:
            self.self_attn = LlamaAttention(config=config, prompt_config=prompt_config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        key_attention_weights: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids_wo_label: Optional[torch.LongTensor] = None,
        attention_mask_flash: Optional[torch.Tensor] = None,
        past_key_attention_weights_v: Optional[torch.Tensor] = None,
        past_key_attention_weights_q: Optional[torch.Tensor] = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            key_attention_weights=key_attention_weights,
            input_ids = input_ids,
            input_ids_wo_label=input_ids_wo_label,
            attention_mask_flash=attention_mask_flash,
            past_key_attention_weights_v = past_key_attention_weights_v,
            past_key_attention_weights_q = past_key_attention_weights_q
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, prompt_config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, prompt_config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.successor=prompt_config['successor']

        self.prompt_config = prompt_config
        self.is_inference = False
        ##########################
        if not prompt_config["run_single"]:
            self.model_dim = config.hidden_size
            self.all_attn_weights = []
        ##########################

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def cal_attention(self, prompt_key, text_input, return_logits=False):
        avg_inputs_embeds = text_input.max(dim=1, keepdim=True).values
        x = self.trans_input(avg_inputs_embeds)
        # x.shape (B,1,D)
        
        if self.prompt_config["attn_temperature"] == 1:
            attn_temperature = math.sqrt(self.model_dim)
        elif self.prompt_config["attn_temperature"] == 2:
            attn_temperature = math.sqrt(self.model_dim / 2)
        else:
            attn_temperature = 1
        
        attn_scores = prompt_key.bmm(
            x.transpose(1, 2)) / attn_temperature
        # attn_scores.shape (B,L,1)
        
        weights = torch.nn.functional.softmax(attn_scores, dim=1)

        if not return_logits:
            return weights  
        else:
            return attn_scores  # shape (B, L, 1)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_ids_wo_label: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )

        if self.prompt_config['flash_attention'] == True:
            if attention_mask is not None and 0.0 in attention_mask:
                attention_mask_flash=attention_mask
            else:
                attention_mask_flash=None
        else:
            attention_mask_flash=None

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        
        #####################
        key_attention_weights = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None


        for idx, decoder_layer in enumerate(self.layers):
            if self.successor is None:
                past_key_attention_weights_v=None
                past_key_attention_weights_q=None
            else:
                if idx in self.successor:
                    past_key_attention_weights_v=None
                    past_key_attention_weights_q=None
                else:
                    past_key_attention_weights_v=self.layers[idx-1].self_attn.key_attention_weights_v
                    past_key_attention_weights_q=self.layers[idx-1].self_attn.key_attention_weights_q

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                raise Exception("gradient_checkpointing is running")
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    key_attention_weights=key_attention_weights,
                    input_ids=input_ids,
                    input_ids_wo_label=input_ids_wo_label,
                    attention_mask_flash=attention_mask_flash,
                    past_key_attention_weights_v=past_key_attention_weights_v,
                    past_key_attention_weights_q=past_key_attention_weights_q
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config, prompt_config):
        super().__init__(config)
        self.model = LlamaModel(config, prompt_config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def memory_replay(self, input_ids, replay_labels):
        kl_loss = None
        if replay_labels is not None:
            
            inputs_embeds = self.model.embed_tokens(input_ids)
            k = input_ids.shape[0]
            kl_loss_fct = nn.KLDivLoss(reduction="batchmean")
            
            pre_prompt_key = torch.cat([self.model.prompt_key.repeat(k, 1, 1), self.model.previous_prompts_keys.repeat(k, 1, 1)], dim=1)
        
            attn_scores = self.model.cal_attention(pre_prompt_key, inputs_embeds, return_logits=True)

            kl_loss = kl_loss_fct(torch.nn.functional.log_softmax(attn_scores.squeeze(2), 1), replay_labels.squeeze().repeat(k, 1))
        return kl_loss

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_ids_wo_label: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_ids_wo_label=input_ids_wo_label,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_ids_wo_label = kwargs.get("input_ids_wo_label", None)

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "input_ids_wo_label": input_ids_wo_label,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )