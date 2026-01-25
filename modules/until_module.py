# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
PyTorch BERT model - Unified version for CLIPKG4Clip and TempMe.

This module contains utility classes and loss functions for both CLIPKG4Clip 
and TempMe implementations.
"""

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from modules.until_config import PretrainedConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Common Utilities (Shared between CLIPKG4Clip and TempMe)
# ============================================================================

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None,  *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model

##########################################################
###### Loss Functions - CLIPKG4Clip Specific #############
##########################################################
class CrossEn(nn.Module):
    """
    Cross Entropy Loss for CLIPKG4Clip (original version without config).
    For TempMe version with config parameter, use CrossEn_TempMe.
    """
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class MILNCELoss(nn.Module):
    def __init__(self, batch_size=1, n_pair=1,):
        super(MILNCELoss, self).__init__()
        self.batch_size = batch_size
        self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix):
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair)))
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (self.n_pair//2)
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=self.bool_dtype)).mean()
        return sim_loss

class MaxMarginRankingLoss(nn.Module):
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()

class AllGather(torch.autograd.Function):
    """
    AllGather for CLIPKG4Clip (original version).
    An autograd function that performs allgather on a tensor.
    For TempMe version with world_size check, use AllGather_TempMe.
    """

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )


##########################################################
########### Loss Functions - TempMe Specific #############
##########################################################

class CrossEn_TempMe(nn.Module):
    """
    Cross Entropy Loss for TempMe (with optional config parameter).
    This is the TempMe version of CrossEn.
    """
    def __init__(self, config=None):
        super(CrossEn_TempMe, self).__init__()
        self.config = config

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class ArcCrossEn(nn.Module):
    """Arc Face Cross Entropy Loss (TempMe)."""
    def __init__(self, margin=10):
        super(ArcCrossEn, self).__init__()
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, sim_matrix, scale):
        cos = torch.diag(sim_matrix)
        sin = torch.sqrt(1.0 - torch.pow(cos, 2))
        pin = cos * self.cos_m - sin * self.sin_m
        sim_matrix = sim_matrix - torch.diag_embed(cos) + torch.diag_embed(pin)
        logpt = F.log_softmax(sim_matrix / scale, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class CrossEn0(nn.Module):
    """Cross Entropy Loss variant 0 (TempMe)."""
    def __init__(self, config=None):
        super(CrossEn0, self).__init__()

    def forward(self, sim_matrix, b):
        logpt = F.log_softmax(sim_matrix[:b, :], dim=-1)
        logpt = torch.diag(logpt[:, :b])
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class ema_CrossEn(nn.Module):
    """EMA Cross Entropy Loss (TempMe)."""
    def __init__(self, config=None):
        super(ema_CrossEn, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        m, n = sim_matrix0.size()
        diag1 = torch.diag(sim_matrix1)
        diag1 = torch.diag_embed(diag1)
        sim_matrix1 = sim_matrix1 - diag1
        logpt = F.log_softmax(torch.cat([sim_matrix0, sim_matrix1], dim=-1), dim=-1)
        logpt = torch.diag(logpt[:, :n])
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class DC_CrossEn(nn.Module):
    """DC Cross Entropy Loss (TempMe)."""
    def __init__(self, config=None):
        super(DC_CrossEn, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1, seta=0.8):
        diag0 = torch.diag(sim_matrix0)
        diag1 = torch.diag(sim_matrix1)
        sim_matrix0 = sim_matrix0 - diag0
        sim_matrix1 = sim_matrix1 - diag1
        m, n = sim_matrix0.size()

        sim_matrix = torch.where(sim_matrix1 < seta, sim_matrix0, torch.tensor(0.0).to(sim_matrix0.device))
        sim_matrix = sim_matrix + diag0

        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class ema_CrossEn1(nn.Module):
    """EMA Cross Entropy Loss variant 1 (TempMe)."""
    def __init__(self, config=None):
        super(ema_CrossEn1, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt0 = F.log_softmax(sim_matrix0, dim=-1)
        logpt1 = F.softmax(sim_matrix1, dim=-1)
        sim_loss = - logpt0 * logpt1
        # diag = torch.diag(sim_loss)
        # sim_loss = sim_loss - diag
        sim_loss = sim_loss.mean()
        return sim_loss


class ema_CrossEn2(nn.Module):
    """EMA Cross Entropy Loss variant 2 (TempMe)."""
    def __init__(self, config=None):
        super(ema_CrossEn2, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1, lambd=0.5):
        m, n = sim_matrix1.size()

        logpt0 = F.log_softmax(sim_matrix0, dim=-1)
        logpt1 = F.softmax(sim_matrix1, dim=-1)
        logpt1 = lambd * torch.eye(m).to(logpt1.device) + (1 - lambd) * logpt1

        sim_loss = - logpt0 * logpt1
        sim_loss = sim_loss.sum() / m
        return sim_loss


class KL(nn.Module):
    """KL Divergence Loss (TempMe)."""
    def __init__(self, config=None):
        super(KL, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt0 = F.log_softmax(sim_matrix0, dim=-1)
        logpt1 = F.softmax(sim_matrix1, dim=-1)
        kl = F.kl_div(logpt0, logpt1, reduction='mean')
        # kl = F.kl_div(logpt0, logpt1, reduction='sum')
        return kl


def _batch_hard(mat_distance, mat_similarity, indice=False):
    """Helper function for triplet loss (TempMe)."""
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1,
                                                       descending=False)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1,
                                                       descending=True)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


class SoftTripletLoss(nn.Module):
    """Soft Triplet Loss (TempMe)."""
    def __init__(self, config=None):
        super(SoftTripletLoss, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        N = sim_matrix0.size(0)
        mat_sim = torch.eye(N).float().to(sim_matrix0.device)
        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(sim_matrix0, mat_sim, indice=True)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        dist_ap_ref = torch.gather(sim_matrix1, 1, ap_idx.view(N, 1).expand(N, N))[:, 0]
        dist_an_ref = torch.gather(sim_matrix1, 1, an_idx.view(N, 1).expand(N, N))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()
        loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        return loss


class MSE(nn.Module):
    """Mean Squared Error Loss (TempMe)."""
    def __init__(self, config=None):
        super(MSE, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt = (sim_matrix0 - sim_matrix1)
        loss = logpt * logpt
        return loss.mean()


def euclidean_dist(x, y):
    """Calculate Euclidean distance matrix (TempMe)."""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def uniformity_loss(x, y):
    """Calculate uniformity loss (TempMe)."""
    input = torch.cat((x, y), dim=0)
    m = input.size(0)
    dist = euclidean_dist(input, input)
    return torch.logsumexp(torch.logsumexp(dist, dim=-1), dim=-1) - torch.log(torch.tensor(m * m - m))


class AllGather_TempMe(torch.autograd.Function):
    """
    AllGather for TempMe (with world_size check).
    An autograd function that performs allgather on a tensor.
    """

    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
        )


class AllGather2(torch.autograd.Function):
    """
    AllGather2 for TempMe (with gradient all_reduce).
    Reference: https://github.com/PyTorchLightning/lightning-bolts
    """

    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        return (grad_input[ctx.rank * ctx.batch_size:(ctx.rank + 1) * ctx.batch_size], None)


# ============================================================================
# Compatibility Aliases and Factory Functions
# ============================================================================

def get_cross_entropy_loss(use_tempme=False, **kwargs):
    """
    Factory function to get appropriate CrossEn loss.
    
    Args:
        use_tempme (bool): If True, return TempMe version with config support
        **kwargs: Arguments passed to loss constructor
        
    Returns:
        CrossEn or CrossEn_TempMe instance
        
    Example:
        # CLIPKG4Clip mode
        loss = get_cross_entropy_loss(use_tempme=False)
        
        # TempMe mode
        loss = get_cross_entropy_loss(use_tempme=True, config=config)
    """
    if use_tempme:
        return CrossEn_TempMe(**kwargs)
    else:
        return CrossEn(**kwargs)


def get_all_gather(use_tempme=False):
    """
    Factory function to get appropriate AllGather.
    
    Args:
        use_tempme (bool): If True, return TempMe version with world_size check
        
    Returns:
        AllGather or AllGather_TempMe class
        
    Example:
        # CLIPKG4Clip mode
        allgather = get_all_gather(use_tempme=False).apply
        
        # TempMe mode  
        allgather = get_all_gather(use_tempme=True).apply
    """
    if use_tempme:
        return AllGather_TempMe
    else:
        return AllGather
