# -*- coding: utf-8 -*-
import torch
import torchvision.utils
from torch import nn
from torchvision.models.resnet import resnet34, resnet50
from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg16
from torchvision.models.resnet import Bottleneck
from network.weight_init import trunc_normal_

from network.pow_models import UNetTiny, ConvEmbedding

import torch.utils.model_zoo as model_zoo
import numpy as np
from utils import ele_max

# Original MSA (Multi-head self-attention)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, vis=False):
        super(Attention, self).__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key   = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.vis = vis

    def forward(self, x):
        B, N, C = x.shape  # N = 1 + N_{patches} (has CLS token) or N_{patches} (don't have CLS token)

        q, k, v = self.query(x), self.key(x), self.value(x)

        new_shape = (B, N, self.num_heads, C // self.num_heads)
        q, k, v = q.reshape(new_shape), k.reshape(new_shape), v.reshape(new_shape)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.reshape(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_probs = attn if self.vis else None  # output attentions
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_probs

class FeedForward(nn.Module):
    """
    Implementation of MLP for transformer
    """

    def __init__(self, dim, hidden_dim, dropout_rate=0.0, revised=False):
        super(FeedForward, self).__init__()
        if not revised:
            """
            Original: https://arxiv.org/pdf/2010.11929.pdf
            """
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, dim),
            )

        else:
            """
            Scaled ReLU: https://arxiv.org/pdf/2109.03810.pdf
            """
            self.net = nn.Sequential(
                nn.Conv1d(dim, hidden_dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Conv1d(hidden_dim, dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(dim),
                nn.GELU(),
            )

        self.revised = revised
        # self._init_weights()
        self.apply(self._init_weights)

    # def _init_weights(self):
    #     for name, module in self.net.named_children():  # or self.net.modules()
    #         if isinstance(module, nn.Linear):
    #             nn.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 # nn.init.normal_(module.bias, std=1e-6)
    #                 # trunc_normal_(module.bias, std=1e-6)
    #                 nn.init.constant_(module.bias, 0)
    #         elif isinstance(module, nn.Conv1d):
    #             nn.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 # nn.init.normal_(module.bias, std=1e-6)
    #                 # trunc_normal_(module.bias, std=1e-6)
    #                 nn.init.constant_(module.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.revised:
            x = x.permute(0, 2, 1)
            x = self.net(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.net(x)

        return x

# Classification layer
class OutputLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes=1000,
        representation_size=None,
        cls_head=False,
    ):
        super(OutputLayer, self).__init__()

        self.num_classes = num_classes
        modules = []
        if representation_size:
            modules.append(nn.Linear(embedding_dim, representation_size))
            modules.append(nn.Tanh())
            modules.append(nn.Linear(representation_size, num_classes))
        else:
            modules.append(nn.Linear(embedding_dim, num_classes))

        self.net = nn.Sequential(*modules)

        if cls_head:
            self.to_cls_token = nn.Identity()

        self.cls_head = cls_head
        self.num_classes = num_classes
        # self._init_weights()

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                if module.weight.shape[0] == self.num_classes:
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.cls_head:
            x = self.to_cls_token(x[:, 0])
        else:
            """
            Scaling Vision Transformer: https://arxiv.org/abs/2106.04560
            """
            x = torch.mean(x, dim=1)

        return self.net(x)


def kernel_rbf(X1: torch.Tensor, X2: torch.Tensor):
    '''
    Pytorch Implementation of computing
    isotropic squared exponential kernel. Computes
    a covariance matrix from points in X1 and X2.

    Args:
        X1: Array of m points (m x d or * x d).
        X2: Array of n points (n x d or * x d).

    Returns:
        Covariance matrix (m x n or * x *).

    Author: Changsheng Lu
    Date: 2020.10.23
    '''
    # shape = X1.shape
    # new_shape = (*shape[:-1], 1)
    # sqdist = torch.sum(X1 ** 2, dim=-1).reshape(new_shape) + torch.sum(X2 ** 2, dim=-1).unsqueeze(dim=-2) - 2 * (X1 @ X2.transpose(-1, -2))
    sqdist = ((X1.unsqueeze(dim=-2) - X2.unsqueeze(dim=-3)) ** 2).sum(dim=-1)
    return sqdist

# Saliency-guided MSA (Multi-head self-attention)
class SalAttention_backup(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, has_CLS=False, vis=False):
        super(SalAttention, self).__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key   = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.has_CLS = has_CLS
        self.vis = vis

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

    def forward(self, x, sal_mask=None, mask_mode=0, att_type='softmax', func_mode=0, beta=1.0, eps=0.3, use_max=False, rbf_l2norm=False, rownorm=False):
        '''
        x: B x N x C, N = 1 + N_{patches} (has CLS token) or N = N_{patches} (don't have CLS token)
        att_type: 'softmax' or 'rbf-inner' or 'rbf-outer'
        sal_mask: B x N_{patches} or None
        mask_mode: 0, 1, 2, namely alpha(M)_{i,j} = M_{i} or (M_{i} + M_{j})/2 or M_{i} * M_{j}
        att_type : 'softmax' or 'rbf'
        beta: 'softmax' attention, if sal_mask is None, beta=1.0;
                                   if sal_mask not None, beta (~ 1.0);
              'rbf' attention, if sal_mask is None, beta=1.0;
                               if sal_mask not None, beta ?
        eps: 0.1, 0.2, 0.3, 0.4, 0.5, ...
        use_max: if true max(beta * alpha(M), eps)
        func_mode: 0, using inverse form 1/(eps + beta * alpha(M)); 1, use (eps + beta * alpha(M))
        '''
        assert beta > 0, 'Error for beta <= 0 in SalAttention.'
        # eps = 0.3

        B, N, C = x.shape  # N = 1 + N_{patches} (has CLS token) or N_{patches} (don't have CLS token)
        if self.has_CLS == True:  # append mask for CLS token
            assert 1 + sal_mask.shape[1] == N and sal_mask.shape[0] == B, "sal_mask should be B x Np where Np is num of patches (exclusive CLS)"
            ones = torch.ones(B, 1).cuda() if torch.cuda.is_available() else torch.ones(B, 1)
            sal_mask = torch.cat([ones, sal_mask], dim=1)  # B x (1 + N_{patches}) = B x N

        q, k, v = self.query(x), self.key(x), self.value(x)

        new_shape = (B, N, self.num_heads, C // self.num_heads)
        q, k, v = q.reshape(new_shape), k.reshape(new_shape), v.reshape(new_shape)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        if att_type == 'softmax':
            if sal_mask is None:
                attn = (q @ k.transpose(-2, -1)) * self.scale * beta
                attn = attn.softmax(dim=-1)
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale
                # attn *= sal_mask.view(B, 1, 1, N)  # apply saliency mask (cannot directly multiply due to negative product)
                if mask_mode == 0:  # alpha(M)_{i,j} = M_{i}
                    sal_mask = sal_mask.view(B, 1, 1, N)
                elif mask_mode == 1:  # alpha(M)_{i,j} = (M_{i} + M_{j})/2
                    sal_mask = (sal_mask.reshape(B, N, 1) + sal_mask.reshape(B, 1, N))/2
                    sal_mask = sal_mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 2:  # alpha(M)_{i,j} = M_{i} * M_{j}
                    sal_mask = (sal_mask.reshape(B, N, 1) * sal_mask.reshape(B, 1, N))
                    sal_mask = sal_mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 3:  # Ones - vec(M)vec(I-M)^{T}
                    sal_mask = sal_mask.reshape(B, N, 1) * (1 - sal_mask.reshape(B, 1, N))
                    sal_mask = (1 - sal_mask).unsqueeze(dim=1)
                else:
                    print('Error for mask mode in SalAttention.')
                    exit(0)

                # ----------------just for debug-------------
                # m = sal_mask.cpu().detach().reshape(B, 1, 1, 12, 12).numpy() if mask_mode == 0 else \
                #     sal_mask.cpu().detach().reshape(B, 1, 144, 12, 12).numpy()
                # a = attn.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # a2 = attn.softmax(dim=-1).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # ----------------just for debug-------------

                if func_mode == 0:  # use inv form
                    sal_sign = (attn <= 0).detach().float()
                    w = (sal_mask*beta + eps) if use_max == False else ele_max(sal_mask*beta, eps)
                    attn = attn * sal_sign / w + attn * (1-sal_sign) * w

                    # # ----------------just for debug-------------
                    # b = attn.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                    # b2 = attn.softmax(dim=-1).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                    # print('ok')
                    # # ----------------just for debug-------------

                    attn = attn.softmax(dim=-1)

                elif func_mode == 1:
                    attn = attn * beta
                    attn = attn.softmax(dim=-1)
                    w = (sal_mask + eps) if use_max == False else ele_max(sal_mask, eps)
                    attn = attn * w

                    if rownorm:
                        rbf_attn_row_sum = (attn.sum(dim=-1) + 1e-9)
                        attn = attn / rbf_attn_row_sum.view(B, self.num_heads, N, 1)

                    # ----------------just for debug-------------
                    # b = attn.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                    # print('ok')
                    # ----------------just for debug-------------
                elif func_mode == 2:  # not suggested
                    sal_sign = (attn <= 0).detach().float()
                    w1 = ((1 - sal_mask) * beta + eps) if use_max == False else ele_max((1 - sal_mask) * beta, eps)
                    w2 = (sal_mask*beta + eps) if use_max == False else ele_max(sal_mask*beta, eps)
                    attn = attn * sal_sign * w1 + attn * (1-sal_sign) * w2
                    attn = attn.softmax(dim=-1)
                else:
                    print('Error for rbf type in SalAttention.')
                    exit(0)

        elif att_type == 'rbf': # att_type == 'rbf-inner' or 'rbf-outer'
            if rbf_l2norm:
                q = torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-12)
                k = torch.nn.functional.normalize(k, p=2, dim=-1, eps=1e-12)
            # l2_dist = torch.sum((q.unsqueeze(dim=3) - k.unsqueeze(dim=2)) ** 2, dim=4)  # B x N_{heads} x Np x Np
            l2_dist = kernel_rbf(q, k)  # B x N_{heads} x Np x Np
            rbf_scale = -0.5 * (self.head_dim ** -0.5) if rbf_l2norm == False else -0.5  # rbf_l2norm has similar effect with 1/sqrt(d)
            if sal_mask is None:
                attn = torch.exp(rbf_scale / beta * l2_dist)
            else:
                if mask_mode == 0:  # alpha(M)_{i,j} = M_{i}
                    sal_mask = sal_mask.view(B, 1, 1, N)
                elif mask_mode == 1:  # alpha(M)_{i,j} = (M_{i} + M_{j})/2
                    sal_mask = (sal_mask.reshape(B, N, 1) + sal_mask.reshape(B, 1, N))/2
                    sal_mask = sal_mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 2:  # alpha(M)_{i,j} = M_{i} * M_{j}
                    sal_mask = (sal_mask.reshape(B, N, 1) * sal_mask.reshape(B, 1, N))
                    sal_mask = sal_mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 3:  # Ones - vec(M)vec(I-M)^{T}
                    sal_mask = sal_mask.reshape(B, N, 1) * (1 - sal_mask.reshape(B, 1, N))
                    sal_mask = (1 - sal_mask).unsqueeze(dim=1)
                else:
                    print('Error for mask_mode in SalAttention.')
                    exit(0)

                #----------------just for debug-------------
                # m = sal_mask.cpu().detach().reshape(B, 1, 1, 12, 12).numpy() if mask_mode == 0 else \
                #     sal_mask.cpu().detach().reshape(B, 1, 144, 12, 12).numpy()
                # l2_dist_temp = l2_dist.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # l2_dist_temp2= (rbf_scale / beta * l2_dist).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # l2_dist_temp3= (rbf_scale * l2_dist / (eps + beta*sal_mask)).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # attn_temp = torch.exp(rbf_scale / beta * l2_dist)
                # a = attn_temp.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # ----------------just for debug-------------


                if func_mode == 0:  # inner case
                    w = (sal_mask*beta + eps) if use_max == False else ele_max(sal_mask*beta, eps)
                    attn = torch.exp(rbf_scale * l2_dist / w)
                elif func_mode == 1:  # outer case
                    attn = torch.exp(rbf_scale / beta * l2_dist)
                    w = (sal_mask + eps) if use_max == False else ele_max(sal_mask, eps)
                    attn = attn * w
                elif func_mode == 2:  # not suggested! Inner case with form ((1 - sal_mask) * beta + eps)
                    w = ((1 - sal_mask) * beta + eps) if use_max == False else ele_max((1 - sal_mask) * beta, eps)
                    attn = torch.exp(rbf_scale * l2_dist * w)
                else:
                    print('Error for rbf type in SalAttention.')
                    exit(0)

                # ----------------just for debug-------------
                # b = attn.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # print('ok')
                # ----------------just for debug-------------

            if rownorm:
                rbf_attn_row_sum = (attn.sum(dim=-1) + 1e-9)
                attn = attn / rbf_attn_row_sum.view(B, self.num_heads, N, 1)
        else:
            print('Error att_type in SalAttention.')
            exit(0)

        attn_probs = attn if self.vis else None  # output attentions
        attn = self.attn_drop(attn)

        # retrieve context from values using attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_probs

def row_normalize(m, type=0):
    '''
    m: tensor
    type: 0, no row_norm; 1, summation for row_norm; 2, softmax for row_norm
    return the normalized matrix whose each row has summation to be 1
    '''
    shape = m.shape
    if type == 1:
        row_sum = (m.sum(dim=-1) + 1e-9)
        m = m / row_sum.view(*shape[0:-1], 1)
    elif type == 2:
        m = m.softmax(dim=-1)

    return m

class SalAttention_backup2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_same=False, attn_drop=0.0, proj_drop=0.0, has_CLS=False, vis=False,
                 att_type='rbf'):
        '''
        # att_pos_enc: it has no effect otherwise att_pos_enc is RPE (relative PE) or CRPE
        '''
        super(SalAttention, self).__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qk_same = qk_same
        if qk_same == False:
            self.query = nn.Linear(dim, dim, bias=qkv_bias)
            self.key   = nn.Linear(dim, dim, bias=qkv_bias)
            self.value = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.query = nn.Linear(dim, dim, bias=qkv_bias)
            # self.key = nn.Linear(dim, dim, bias=qkv_bias)
            self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.has_CLS = has_CLS
        self.vis = vis
        self.att_type = att_type  # 'softmax', 'rbf', 'softmax-fg'/'rbf-fg'

        if self.att_type in ['softmax-fg', 'rbf-fg', 'softmax-fg2', 'rbf-fg2', 'softmax-fg3', 'rbf-fg3']:
            self.bg = nn.Linear(dim, dim, bias=qkv_bias)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

    def forward(self, x, sal_mask=None, mask_mode=0, func_mode=0, beta=1.0, eps=0.3, thresh=0.5, use_max=False, rbf_l2norm=False, rownorm=0):
        '''
        x: B x N x C, N = 1 + N_{patches} (has CLS token) or N = N_{patches} (don't have CLS token)
        sal_mask: B x N_{patches} or None
        mask_mode: 0, 1, 2, namely alpha(M)_{i,j} = M_{i} or (M_{i} + M_{j})/2 or M_{i} * M_{j}
        att_type : 'rbf', 'softmax', 'rbf_fg'/'softmax_fg'
        beta: 'softmax' attention, if sal_mask is None, beta=1.0;
                                   if sal_mask not None, beta (~ 1.0);
              'rbf' attention, if sal_mask is None, beta=1.0;
                               if sal_mask not None, beta ?
        eps: 0.1, 0.2, 0.3, 0.4, 0.5, ...
        use_max: if true max(beta * alpha(M), eps)
        func_mode: 0, using inverse form 1/(eps + beta * alpha(M)); 1, use (eps + beta * alpha(M))
        '''
        assert beta > 0, 'Error for beta <= 0 in SalAttention.'
        # eps = 0.3

        B, N, C = x.shape  # N = 1 + N_{patches} (has CLS token) or N_{patches} (don't have CLS token)
        if self.has_CLS == True:  # append mask for CLS token
            assert 1 + sal_mask.shape[1] == N and sal_mask.shape[0] == B, "sal_mask should be B x Np where Np is num of patches (exclusive CLS)"
            ones = torch.ones(B, 1).cuda() if torch.cuda.is_available() else torch.ones(B, 1)
            sal_mask = torch.cat([ones, sal_mask], dim=1)  # B x (1 + N_{patches}) = B x N

        # q, k different or same
        if self.qk_same == False:
            if self.att_type in ['softmax', 'rbf']:
                q, k, v = self.query(x), self.key(x), self.value(x)
            elif self.att_type in ['softmax-fg', 'rbf-fg']:
                q = self.query(x * sal_mask.view(B, N, 1))
                k, v = self.key(x), self.value(x)
            elif self.att_type in ['softmax-fg2', 'rbf-fg2']:
                q = self.query(x * sal_mask.view(B, N, 1))
                k = self.key(x * sal_mask.view(B, N, 1))
                v = self.value(x)
            elif self.att_type in ['softmax-fg3', 'rbf-fg3']:
                q = self.query(x * sal_mask.view(B, N, 1))
                k = self.key(x * sal_mask.view(B, N, 1))
                v = self.value(x * sal_mask.view(B, N, 1))

            new_shape = (B, N, self.num_heads, C // self.num_heads)
            q, k, v = q.reshape(new_shape), k.reshape(new_shape), v.reshape(new_shape)
            q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        else:  # q, k are same
            if self.att_type in ['softmax', 'rbf']:
                q, v = self.query(x), self.value(x)
            elif self.att_type in ['softmax-fg', 'rbf-fg', 'softmax-fg2', 'rbf-fg2']:
                q = self.query(x * sal_mask.view(B, N, 1))
                v = self.value(x)
            elif self.att_type in ['softmax-fg3', 'rbf-fg3']:
                q = self.query(x * sal_mask.view(B, N, 1))
                v = self.value(x * sal_mask.view(B, N, 1))

            new_shape = (B, N, self.num_heads, C // self.num_heads)
            q, v = q.reshape(new_shape), v.reshape(new_shape)
            q, v = q.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
            k = q * 1.0  # this is to avoid changing k when q is modified.

        # whether automatically learn dilation

        if self.att_type == 'softmax':
            if sal_mask is None:
                attn = (q @ k.transpose(-2, -1)) * self.scale * beta
                attn = attn.softmax(dim=-1)
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale
                # attn *= sal_mask.view(B, 1, 1, N)  # apply saliency mask (cannot directly multiply due to negative product)
                if mask_mode == 0:  # alpha(M)_{i,j} = M_{i}
                    mask = sal_mask.view(B, 1, 1, N)
                elif mask_mode == 1:  # alpha(M)_{i,j} = (M_{i} + M_{j})/2
                    mask = (sal_mask.reshape(B, N, 1) + sal_mask.reshape(B, 1, N))/2
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 2:  # alpha(M)_{i,j} = M_{i} * M_{j}
                    mask = (sal_mask.reshape(B, N, 1) * sal_mask.reshape(B, 1, N))
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 3:  # Ones - vec(M)vec(I-M)^{T}
                    mask = sal_mask.reshape(B, N, 1) * (1 - sal_mask.reshape(B, 1, N))
                    mask = (1 - mask).unsqueeze(dim=1)
                else:
                    print('Error for mask mode in SalAttention.')
                    exit(0)

                # ----------------just for debug-------------
                # m = sal_mask.cpu().detach().reshape(B, 1, 1, 12, 12).numpy() if mask_mode == 0 else \
                #     sal_mask.cpu().detach().reshape(B, 1, 144, 12, 12).numpy()
                # a = attn.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # a2 = attn.softmax(dim=-1).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # ----------------just for debug-------------

                if func_mode == 0:  # use inv form
                    sal_sign = (attn <= 0).detach().float()
                    w = (mask*beta) if use_max == False else ele_max(mask*beta, eps)
                    attn = attn * sal_sign / w + attn * (1-sal_sign) * w

                    # # ----------------just for debug-------------
                    # b = attn.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                    # b2 = attn.softmax(dim=-1).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                    # print('ok')
                    # # ----------------just for debug-------------

                    attn = attn.softmax(dim=-1)

                elif func_mode == 1:
                    attn = attn * beta
                    attn = attn.softmax(dim=-1)
                    w = (mask) if use_max == False else ele_max(mask, eps)
                    attn = attn * w

                    if rownorm:
                        attn = row_normalize(attn, rownorm)

                    # ----------------just for debug-------------
                    # b = attn.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                    # print('ok')
                    # ----------------just for debug-------------
                elif func_mode == 2:  # not suggested
                    sal_sign = (attn <= 0).detach().float()
                    w1 = ((1 - mask) * beta + eps) if use_max == False else ele_max((1 - mask) * beta, eps)
                    w2 = (mask*beta) if use_max == False else ele_max(mask*beta, eps)
                    attn = attn * sal_sign * w1 + attn * (1-sal_sign) * w2
                    attn = attn.softmax(dim=-1)
                else:
                    print('Error for rbf type in SalAttention.')
                    exit(0)
            attn_probs = attn if self.vis else None  # output attentions
            attn = self.attn_drop(attn)
            # retrieve context from values using attention
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif self.att_type == 'rbf':
            if rbf_l2norm:
                q = torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-12)
                k = torch.nn.functional.normalize(k, p=2, dim=-1, eps=1e-12)
            # l2_dist = torch.sum((q.unsqueeze(dim=3) - k.unsqueeze(dim=2)) ** 2, dim=4)  # B x N_{heads} x Np x Np
            l2_dist = kernel_rbf(q, k)  # B x N_{heads} x Np x Np
            rbf_scale = -0.5 * (self.head_dim ** -0.5) if rbf_l2norm == False else -0.5  # rbf_l2norm has similar effect with 1/sqrt(d)
            if sal_mask is None:
                attn = torch.exp(rbf_scale / beta * l2_dist)
            else:
                if mask_mode == 0:  # alpha(M)_{i,j} = M_{i}
                    mask = sal_mask.view(B, 1, 1, N)
                elif mask_mode == 1:  # alpha(M)_{i,j} = (M_{i} + M_{j})/2
                    mask = (sal_mask.reshape(B, N, 1) + sal_mask.reshape(B, 1, N))/2
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 2:  # alpha(M)_{i,j} = M_{i} * M_{j}
                    mask = (sal_mask.reshape(B, N, 1) * sal_mask.reshape(B, 1, N))
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 3:  # Ones - vec(M)vec(I-M)^{T}
                    mask = sal_mask.reshape(B, N, 1) * (1 - sal_mask.reshape(B, 1, N))
                    mask = (1 - mask).unsqueeze(dim=1)
                elif mask_mode == 4:  # vec(M)*vec(M)^{T} + (M < fg_thresh)*I (same to I-M)
                    # fg_thresh = thresh  # 0.5
                    # bg_sign = (sal_mask < fg_thresh).detach().long()  # B x N
                    bg_sign = 1 - sal_mask  # B x N
                    bg_identity = torch.diag_embed(bg_sign)  # B x N x N
                    mask = (sal_mask.reshape(B, N, 1) * sal_mask.reshape(B, 1, N))  # B x N x N
                    mask = mask + bg_identity  # bg weight will exceed 1 but no matters
                    bg_mask_clamp = False
                    if bg_mask_clamp:
                        mask = torch.clamp(mask, max=1.0)
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                else:
                    print('Error for mask_mode in SalAttention.')
                    exit(0)

                #----------------just for debug-------------
                # m = sal_mask.cpu().detach().reshape(B, 1, 1, 12, 12).numpy() if mask_mode == 0 else \
                #     sal_mask.cpu().detach().reshape(B, 1, 144, 12, 12).numpy()
                # l2_dist_temp = l2_dist.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # l2_dist_temp2= (rbf_scale / beta * l2_dist).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # l2_dist_temp3= (rbf_scale * l2_dist / (eps + beta*sal_mask)).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # attn_temp = torch.exp(rbf_scale / beta * l2_dist)
                # a = attn_temp.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # ----------------just for debug-------------


                if func_mode == 0:  # inner case
                    w = (mask*beta) if use_max == False else ele_max(mask*beta, eps)
                    attn = torch.exp(rbf_scale * l2_dist / w)
                elif func_mode == 1:  # outer case
                    attn = torch.exp(rbf_scale / beta * l2_dist)
                    w = (mask) if use_max == False else ele_max(mask, eps)
                    attn = attn * w
                elif func_mode == 2:  # not suggested! Inner case with form ((1 - mask) * beta + eps)
                    w = ((1 - mask) * beta + eps) if use_max == False else ele_max((1 - mask) * beta, eps)
                    attn = torch.exp(rbf_scale * l2_dist * w)
                elif func_mode == 3:  # outer case + bg_identity ((M < fg_thresh)*I, same to I-M)
                    # fg_thresh = thresh  # 0.5
                    # bg_sign = (sal_mask < fg_thresh).detach().long()  # B x N
                    bg_sign = 1 - sal_mask  # B x N
                    bg_identity = torch.diag_embed(bg_sign)  # B x N x N
                    # bg_mask_clamp = False
                    # mask = mask + bg_identity.unsqueeze(dim=1)  # bg weight will exceed 1 but no matters
                    # if bg_mask_clamp:
                    #     mask = torch.clamp(mask, max=1.0)

                    attn = torch.exp(rbf_scale / beta * l2_dist)
                    w = (mask) if use_max == False else ele_max(mask, eps)
                    attn = attn * w + bg_identity.unsqueeze(dim=1)  # attn value may exceed 1 but should not matter much
                elif func_mode == 4:  # addition, (attn + mask) akin to self-attn + GNN
                    attn = torch.exp(rbf_scale / beta * l2_dist)
                    w = (mask) if use_max == False else ele_max(mask, eps)
                    attn = (attn + w) # / 2
                else:
                    print('Error for rbf type in SalAttention.')
                    exit(0)

                # ----------------just for debug-------------
                # b = attn.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # print('ok')
                # ----------------just for debug-------------

            if rownorm:  # we could also use softmax to perform row_norm
                attn = row_normalize(attn, rownorm)

            attn_probs = attn if self.vis else None  # output attentions
            attn = self.attn_drop(attn)
            # retrieve context from values using attention
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif self.att_type in ['softmax-fg', 'rbf-fg', 'softmax-fg2', 'rbf-fg2', 'softmax-fg3', 'rbf-fg3']:
            assert sal_mask is not None, 'Foreground needs sal_mask.'
            # q = q * sal_mask.view(B, 1, N, 1)  # B x N_h x N_p x d
            # v_bg = (1 - sal_mask).view(B, N, 1) * self.bg(x)  # B x N x C
            v_bg = self.bg(x * (1 - sal_mask).view(B, N, 1))  # B x N x C
            if self.att_type in ['softmax-fg', 'softmax-fg2', 'softmax-fg3']:
                attn = (q @ k.transpose(-2, -1)) * self.scale * beta
                attn = attn.softmax(dim=-1)
            elif self.att_type in ['rbf-fg', 'rbf-fg2', 'rbf-fg3']:
                if rbf_l2norm:
                    q = torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-12)
                    k = torch.nn.functional.normalize(k, p=2, dim=-1, eps=1e-12)
                l2_dist = kernel_rbf(q, k)  # B x N_{heads} x Np x Np
                rbf_scale = -0.5 * (self.head_dim ** -0.5) if rbf_l2norm == False else -0.5  # rbf_l2norm has similar effect with 1/sqrt(d)
                attn = torch.exp(rbf_scale / beta * l2_dist)
                if rownorm:
                    attn = row_normalize(attn, rownorm)

            attn_probs = attn if self.vis else None  # output attentions
            attn = self.attn_drop(attn)
            # retrieve context from values using attention
            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + v_bg
        else:
            print('Error att_type in SalAttention.')
            exit(0)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_probs

class SalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_same=False, attn_drop=0.0, proj_drop=0.0, has_CLS=False, vis=False,
                 att_type='rbf', att_pos_enc='', grid_size=(12,12)):
        '''
        # att_pos_enc: it has no effect otherwise att_pos_enc is RPE (relative PE) or CRPE
        '''
        super(SalAttention, self).__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(d)
        self.rbf_scale = -0.5 * (self.head_dim ** -0.5)  # - 1/ (2 * sqrt(d))

        self.qk_same = qk_same
        if qk_same == False:
            self.query = nn.Linear(dim, dim, bias=qkv_bias)
            self.key   = nn.Linear(dim, dim, bias=qkv_bias)
            self.value = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.query = nn.Linear(dim, dim, bias=qkv_bias)
            # self.key = nn.Linear(dim, dim, bias=qkv_bias)
            self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.has_CLS = has_CLS
        self.vis = vis
        self.att_type = att_type  # 'softmax', 'rbf', 'softmax-fg'/'rbf-fg'

        if self.att_type in ['softmax-fg', 'rbf-fg', 'softmax-fg2', 'rbf-fg2', 'softmax-fg3', 'rbf-fg3']:
            self.bg = nn.Linear(dim, dim, bias=qkv_bias)

        self.apply(self._init_weights)

        # relative PE (RPE) or conditional relative PE (CRPE)
        if att_pos_enc not in ['RPE2', 'RPE4']:
            att_pos_enc = ''  # No RPE
        self.att_pos_enc = att_pos_enc
        self.grid_size = grid_size
        if att_pos_enc == '':  # no RPE
            pass
        elif att_pos_enc == 'RPE2':  # RPE ((2H-1)+(2W-1)) same as to Attention Augmented CNN
            # feature vector; shared across heads
            self.relative_position_repres_h = nn.Parameter(
                torch.zeros((2 * grid_size[0] - 1), self.head_dim))
            self.relative_position_repres_w = nn.Parameter(
                torch.zeros((2 * grid_size[1] - 1), self.head_dim))

            coords_h = torch.arange(grid_size[0])
            coords_w = torch.arange(grid_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += grid_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += grid_size[1] - 1
            self.register_buffer("relative_position_index_h", relative_coords[:, :, 0])  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index_w", relative_coords[:, :, 1])  # Wh*Ww, Wh*Ww
            # self.relative_position_index_h = (relative_coords[:, :, 0]).clone()
            # self.relative_position_index_w = (relative_coords[:, :, 1]).clone()

        elif att_pos_enc == 'RPE4':  # RPE ((2H-1)*(2W-1)) same to Swin-T, treated as bias scalar
            # bias scalr; not shared across heads
            self.relative_position_repres = nn.Parameter(
                torch.zeros((2 * grid_size[0] - 1) * (2 * grid_size[1] -1), num_heads))

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(grid_size[0])
            coords_w = torch.arange(grid_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += grid_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += grid_size[1] - 1
            relative_coords[:, :, 0] *= (2 * grid_size[1] - 1)
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            # self.relative_position_index = relative_position_index
        else:
            raise NotImplementedError

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def attention_logit_compute(self, q, k, beta, att_type='rbf', rbf_l2norm=False):
        '''
        Compute attention logit with or without RPE.
            If attention type is 'softmax', the attention logit is qk^{T}/ (beta * sqrt(d));
            If attention type is 'rbf', the attention logit is -||q_i - k_j||^2 / (2 * beta * sqrt(d))
        q: B x N_{head} x N_{p} x d
        k: B x N_{head} x N_{p} x d
        beta: temperature
        att_type: 'rbf' or 'softmax'
        '''
        B, Nh, Np, d = q.shape
        sm_scale = self.scale / beta
        # rbf_l2norm has similar effect with 1/sqrt(d), thus we can simply use -1/(2*beta)
        rbf_scale = (self.rbf_scale / beta) if rbf_l2norm == False else (-0.5 / beta)
        if self.att_pos_enc == '':  # vanilla self-attention, no RPE
            if att_type == 'softmax':
                attn_logit = (q @ k.transpose(-2, -1)) * sm_scale  # B x Nh x Np x Np
            elif att_type == 'rbf':
                attn_logit = kernel_rbf(q, k) * rbf_scale  # B x Nh x Np x Np
            else:
                raise NotImplementedError
        elif self.att_pos_enc == 'RPE2':  # RPE ((2H-1)+(2W-1)) same as to Attention Augmented CNN
            # retrieve RPE height and RPE width
            assert self.grid_size[0] * self.grid_size[1] == Np
            rpe_vector_h = self.relative_position_repres_h[self.relative_position_index_h.view(-1)].view(
                Np, Np, -1)  # Wh*Ww,Wh*Ww, d
            rpe_vector_w = self.relative_position_repres_w[self.relative_position_index_w.view(-1)].view(
                Np, Np, -1)  # Wh*Ww,Wh*Ww, d
            rpe_vector_h = rpe_vector_h.reshape(1, 1, Np, Np, d)
            rpe_vector_w = rpe_vector_w.reshape(1, 1, Np, Np, d)
            if att_type == 'softmax':
                # QK^{T} + Q*RPE_h + Q*RPE_w
                attn_logit = q @ k.transpose(-2, -1)  # B x Nh x Np x Np
                spatial_logit = (q.unsqueeze(-2) * rpe_vector_h).sum(-1) \
                                + (q.unsqueeze(-2) * rpe_vector_w).sum(-1)  # B x Nh x Np x Np
                attn_logit = (attn_logit + spatial_logit) * sm_scale
            elif att_type == 'rbf':
                attn_logit = ((q.unsqueeze(dim=-2) - (k.unsqueeze(dim=-3) +  rpe_vector_h + rpe_vector_w)) ** 2).sum(dim=-1)
                attn_logit = attn_logit * rbf_scale  # B x Nh x Np x Np
            else:
                raise NotImplementedError
        elif self.att_pos_enc == 'RPE4':  # RPE ((2H-1)*(2W-1)) same to Swin-T, treated as bias scalar
            # retrieve RPE
            assert self.grid_size[0] * self.grid_size[1] == Np
            relative_position_bias = self.relative_position_repres[self.relative_position_index.view(-1)].view(
                Np, Np, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

            if att_type == 'softmax':
                attn_logit = (q @ k.transpose(-2, -1)) * sm_scale  # B x Nh x Np x Np
                attn_logit = attn_logit + relative_position_bias.unsqueeze(0)
            elif att_type == 'rbf':
                attn_logit = kernel_rbf(q, k) * rbf_scale  # B x Nh x Np x Np
                attn_logit = attn_logit - nn.functional.gelu(relative_position_bias.unsqueeze(0))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return attn_logit  # B x Nh x Np x Np

    def forward(self, x, sal_mask=None, mask_mode=0, func_mode=0, beta=1.0, eps=0.3, thresh=0.5, use_max=False, rbf_l2norm=False, rownorm=0):
        '''
        x: B x N x C, N = 1 + N_{patches} (has CLS token) or N = N_{patches} (don't have CLS token)
        sal_mask: B x N_{patches} or None
        mask_mode: 0, 1, 2, namely alpha(M)_{i,j} = M_{i} or (M_{i} + M_{j})/2 or M_{i} * M_{j}
        att_type : 'rbf', 'softmax', 'rbf_fg'/'softmax_fg'
        beta: 'softmax' attention, if sal_mask is None, beta=1.0;
                                   if sal_mask not None, beta (~ 1.0);
              'rbf' attention, if sal_mask is None, beta=1.0;
                               if sal_mask not None, beta ?
        eps: 0.1, 0.2, 0.3, 0.4, 0.5, ...
        use_max: if true max(beta * alpha(M), eps)
        func_mode: 0, using inverse form 1/(eps + beta * alpha(M)); 1, use (eps + beta * alpha(M))
        '''
        assert beta > 0, 'Error for beta <= 0 in SalAttention.'
        # eps = 0.3

        B, N, C = x.shape  # N = 1 + N_{patches} (has CLS token) or N_{patches} (don't have CLS token)
        if self.has_CLS == True:  # append mask for CLS token
            assert 1 + sal_mask.shape[1] == N and sal_mask.shape[0] == B, "sal_mask should be B x Np where Np is num of patches (exclusive CLS)"
            ones = torch.ones(B, 1).cuda() if torch.cuda.is_available() else torch.ones(B, 1)
            sal_mask = torch.cat([ones, sal_mask], dim=1)  # B x (1 + N_{patches}) = B x N

        # q, k different or same
        if self.qk_same == False:
            if self.att_type in ['softmax', 'rbf']:
                q, k, v = self.query(x), self.key(x), self.value(x)
            elif self.att_type in ['softmax-fg', 'rbf-fg']:
                q = self.query(x * sal_mask.view(B, N, 1))
                k, v = self.key(x), self.value(x)
            elif self.att_type in ['softmax-fg2', 'rbf-fg2']:
                q = self.query(x * sal_mask.view(B, N, 1))
                k = self.key(x * sal_mask.view(B, N, 1))
                v = self.value(x)
            elif self.att_type in ['softmax-fg3', 'rbf-fg3']:
                q = self.query(x * sal_mask.view(B, N, 1))
                k = self.key(x * sal_mask.view(B, N, 1))
                v = self.value(x * sal_mask.view(B, N, 1))

            new_shape = (B, N, self.num_heads, C // self.num_heads)
            q, k, v = q.reshape(new_shape), k.reshape(new_shape), v.reshape(new_shape)
            q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        else:  # q, k are same
            if self.att_type in ['softmax', 'rbf']:
                q, v = self.query(x), self.value(x)
            elif self.att_type in ['softmax-fg', 'rbf-fg', 'softmax-fg2', 'rbf-fg2']:
                q = self.query(x * sal_mask.view(B, N, 1))
                v = self.value(x)
            elif self.att_type in ['softmax-fg3', 'rbf-fg3']:
                q = self.query(x * sal_mask.view(B, N, 1))
                v = self.value(x * sal_mask.view(B, N, 1))

            new_shape = (B, N, self.num_heads, C // self.num_heads)
            q, v = q.reshape(new_shape), v.reshape(new_shape)
            q, v = q.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
            k = q * 1.0  # this is to avoid changing k when q is modified.

        # whether automatically learn dilation

        if self.att_type == 'softmax':
            # B x Nh x Np x Np
            attn_logit = self.attention_logit_compute(q, k, beta=beta, att_type='softmax')
            attn_probs_before_calibrate = attn_logit.softmax(dim=-1).clone() if self.vis else None  # B x Nh x Np x Np
            if sal_mask is None:
                attn = attn_logit.softmax(dim=-1)  # B x Nh x Np x Np
            else:
                # attn *= sal_mask.view(B, 1, 1, N)  # apply saliency mask (cannot directly multiply due to negative product)
                if mask_mode == 0:  # alpha(M)_{i,j} = M_{i} * M_{j}
                    mask = (sal_mask.reshape(B, N, 1) * sal_mask.reshape(B, 1, N))
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 1:  # alpha(M)_{i,j} = 2 * M_{i} * M_{j} / (M_{i} + M_{j} + 1e-9)
                    mask = 2 * (sal_mask.reshape(B, N, 1) * sal_mask.reshape(B, 1, N)) / \
                           (sal_mask.reshape(B, N, 1) + sal_mask.reshape(B, 1, N) + 1e-9)
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 2:  # alpha(M)_{i,j} = (M_{i} + M_{j})/2
                    mask = (sal_mask.reshape(B, N, 1) + sal_mask.reshape(B, 1, N)) / 2
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 3:  # Ones - vec(M)vec(I-M)^{T}
                    mask = sal_mask.reshape(B, N, 1) * (1 - sal_mask.reshape(B, 1, N))
                    mask = (1 - mask).unsqueeze(dim=1)
                elif mask_mode == 4:  # alpha(M)_{i,j} = M_{i}
                    mask = sal_mask.view(B, 1, 1, N)
                else:
                    print('Error for mask mode in SalAttention.')
                    exit(0)

                # ----------------just for debug-------------
                # m = sal_mask.cpu().detach().reshape(B, 1, 1, 12, 12).numpy() if mask_mode == 0 else \
                #     sal_mask.cpu().detach().reshape(B, 1, 144, 12, 12).numpy()
                # a = attn.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # a2 = attn.softmax(dim=-1).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # ----------------just for debug-------------

                if func_mode == 0:  # outter case, (A Odot MaskMat) + (I - diag(M)); bg_identity ((M < fg_thresh)*I, same to I-M)
                    # fg_thresh = thresh  # 0.5
                    # bg_sign = (sal_mask < fg_thresh).detach().long()  # B x N
                    bg_sign = 1 - sal_mask  # B x N
                    bg_identity = torch.diag_embed(bg_sign)  # B x N x N
                    # bg_mask_clamp = False
                    # mask = mask + bg_identity.unsqueeze(dim=1)  # bg weight will exceed 1 but no matters
                    # if bg_mask_clamp:
                    #     mask = torch.clamp(mask, max=1.0)

                    # attn = attn_logit.softmax(dim=-1)
                    w = (mask) if use_max == False else ele_max(mask, eps)
                    # attn = attn * w + bg_identity.unsqueeze(dim=1)  # attn value may exceed 1 but should not matter much
                    #
                    # if rownorm:  # we could also use softmax to perform row_norm
                    #     attn = row_normalize(attn, rownorm)

                    z_max, _ = torch.max(attn_logit, dim=3, keepdim=True)  # B x Nh x Np x 1
                    attn_logit = attn_logit - z_max  # make value <=0, avoid value to be too large (otherwise exp(v) will be +inf)
                    attn = torch.exp(attn_logit) * w + bg_identity.unsqueeze(dim=1)
                    attn = row_normalize(attn, type=1)
                    # if torch.sum(torch.isnan(attn.cpu().detach()) == True) > 0:
                    #     print('ok')
                elif func_mode == 1:  # inner case, softmax(A_logit - (1 - (MaskMat + I-diag(M)))*J)
                    bg_sign = 1 - sal_mask  # B x N
                    bg_identity = torch.diag_embed(bg_sign)  # B x N x N
                    mask = mask + bg_identity.unsqueeze(dim=1)  # (MaskMat + I-diag(M))

                    if thresh >= 0:  # fixed thresh
                        attn_logit = attn_logit - (1 - mask) * thresh  # serve as bias to perform masking
                    else:  # adaptive thresh
                        attn_logit_max = attn_logit.detach().max(dim=-1)[0]  # B x Nh x Np
                        attn_logit_min = attn_logit.detach().min(dim=-1)[0]  # B x Nh x Np
                        # attn_logit_mean = attn_logit.detach().mean(dim=-1)[0]  # B x Nh x Np
                        thresh_t = ((attn_logit_max - attn_logit_min) * (-thresh)).unsqueeze(-1)  # B x Nh x Np x 1
                        attn_logit = attn_logit - (1 - mask) * thresh_t

                    attn = attn_logit.softmax(dim=-1)
                elif func_mode == 2:  # additional, A + (MaskMat + I-diag(M)) akin to self-attn + GNN
                    bg_sign = 1 - sal_mask  # B x N
                    bg_identity = torch.diag_embed(bg_sign)  # B x N x N
                    mask = mask + bg_identity.unsqueeze(dim=1)  # (MaskMat + I-diag(M))

                    attn = attn_logit.softmax(dim=-1)
                    w = row_normalize(mask, type=1)
                    attn = (attn + w) / 2  # row summation equal to 1

                    if rownorm:  # we could also use softmax to perform row_norm
                        attn = row_normalize(attn, rownorm)
                else:
                    print('Error for rbf type in SalAttention.')
                    exit(0)
            attn_probs = attn.clone() if self.vis else None  # output attentions
            attn = self.attn_drop(attn)
            # retrieve context from values using attention
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif self.att_type == 'rbf':
            if rbf_l2norm:
                q = torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-12)
                k = torch.nn.functional.normalize(k, p=2, dim=-1, eps=1e-12)
            # B x Nh x Np x Np
            attn_logit = self.attention_logit_compute(q, k, beta=beta, att_type='rbf', rbf_l2norm=rbf_l2norm)
            attn_probs_before_calibrate = torch.exp(attn_logit).clone() if self.vis else None
            if sal_mask is None:
                attn = torch.exp(attn_logit)
            else:
                if mask_mode == 0:  # alpha(M)_{i,j} = M_{i} * M_{j}
                    mask = (sal_mask.reshape(B, N, 1) * sal_mask.reshape(B, 1, N))
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 1:  # alpha(M)_{i,j} = 2 * M_{i} * M_{j} / (M_{i} + M_{j} + 1e-9)
                    mask = 2 * (sal_mask.reshape(B, N, 1) * sal_mask.reshape(B, 1, N)) / \
                           (sal_mask.reshape(B, N, 1) + sal_mask.reshape(B, 1, N) + 1e-9)
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 2:  # alpha(M)_{i,j} = (M_{i} + M_{j})/2
                    mask = (sal_mask.reshape(B, N, 1) + sal_mask.reshape(B, 1, N)) / 2
                    mask = mask.unsqueeze(dim=1)  # B x 1 x N x N
                elif mask_mode == 3:  # Ones - vec(M)vec(I-M)^{T}
                    mask = sal_mask.reshape(B, N, 1) * (1 - sal_mask.reshape(B, 1, N))
                    mask = (1 - mask).unsqueeze(dim=1)
                elif mask_mode == 4:  # alpha(M)_{i,j} = M_{i}
                    mask = sal_mask.view(B, 1, 1, N)
                else:
                    print('Error for mask mode in SalAttention.')
                    exit(0)

                #----------------just for debug-------------
                # m = sal_mask.cpu().detach().reshape(B, 1, 1, 12, 12).numpy() if mask_mode == 0 else \
                #     sal_mask.cpu().detach().reshape(B, 1, 144, 12, 12).numpy()
                # l2_dist_temp = l2_dist.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # l2_dist_temp2= (rbf_scale / beta * l2_dist).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # l2_dist_temp3= (rbf_scale * l2_dist / (eps + beta*sal_mask)).cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # attn_temp = torch.exp(rbf_scale / beta * l2_dist)
                # a = attn_temp.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # ----------------just for debug-------------

                if func_mode == 0:  # outter case, (A Odot MaskMat) + (I - diag(M)); bg_identity ((M < fg_thresh)*I, same to I-M)
                    # fg_thresh = thresh  # 0.5
                    # bg_sign = (sal_mask < fg_thresh).detach().long()  # B x N
                    bg_sign = 1 - sal_mask  # B x N
                    bg_identity = torch.diag_embed(bg_sign)  # B x N x N
                    # bg_mask_clamp = False
                    # mask = mask + bg_identity.unsqueeze(dim=1)  # bg weight will exceed 1 but no matters
                    # if bg_mask_clamp:
                    #     mask = torch.clamp(mask, max=1.0)

                    attn = torch.exp(attn_logit)
                    w = (mask) if use_max == False else ele_max(mask, eps)
                    attn = attn * w + bg_identity.unsqueeze(dim=1)  # attn value may exceed 1 but should not matter much
                elif func_mode == 1:  # inner case, exp(A_logit - (1 - (MaskMat + I-diag(M)))*J)
                    bg_sign = 1 - sal_mask  # B x N
                    bg_identity = torch.diag_embed(bg_sign)  # B x N x N
                    mask = mask + bg_identity.unsqueeze(dim=1)  # (MaskMat + I-diag(M))

                    if thresh >= 0:  # fixed thresh
                        attn_logit = attn_logit - (1 - mask) * thresh  # serve as bias to perform masking
                    else:  # adaptive thresh
                        attn_logit_max = attn_logit.detach().max(dim=-1)[0]  # B x Nh x Np
                        attn_logit_min = attn_logit.detach().min(dim=-1)[0]  # B x Nh x Np
                        # attn_logit_mean = attn_logit.detach().mean(dim=-1)[0]  # B x Nh x Np
                        thresh_t = ((attn_logit_max - attn_logit_min) * (-thresh)).unsqueeze(-1)  # B x Nh x Np x 1
                        attn_logit = attn_logit - (1 - mask) * thresh_t

                    attn = torch.exp(attn_logit)
                elif func_mode == 2:  # additional, A + (MaskMat + I-diag(M)) akin to self-attn + GNN
                    bg_sign = 1 - sal_mask  # B x N
                    bg_identity = torch.diag_embed(bg_sign)  # B x N x N
                    mask = mask + bg_identity.unsqueeze(dim=1)  # (MaskMat + I-diag(M))

                    attn = torch.exp(attn_logit)
                    w = mask
                    attn = (attn + w) / 2
                else:
                    print('Error for rbf type in SalAttention.')
                    exit(0)

                # ----------------just for debug-------------
                # b = attn.cpu().detach().numpy().reshape(-1, self.num_heads, 144, 12, 12)
                # print('ok')
                # ----------------just for debug-------------

            if rownorm:  # we could also use softmax to perform row_norm
                attn = row_normalize(attn, rownorm)

            attn_probs = attn.clone() if self.vis else None  # output attentions
            attn = self.attn_drop(attn)
            # retrieve context from values using attention
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif self.att_type in ['softmax-fg', 'rbf-fg', 'softmax-fg2', 'rbf-fg2', 'softmax-fg3', 'rbf-fg3']:
            assert sal_mask is not None, 'Foreground needs sal_mask.'
            # q = q * sal_mask.view(B, 1, N, 1)  # B x N_h x N_p x d
            # v_bg = (1 - sal_mask).view(B, N, 1) * self.bg(x)  # B x N x C
            v_bg = self.bg(x * (1 - sal_mask).view(B, N, 1))  # B x N x C
            if self.att_type in ['softmax-fg', 'softmax-fg2', 'softmax-fg3']:
                attn_logit = self.attention_logit_compute(q, k, beta, att_type='softmax')
                attn = attn_logit.softmax(dim=-1)
            elif self.att_type in ['rbf-fg', 'rbf-fg2', 'rbf-fg3']:
                if rbf_l2norm:
                    q = torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-12)
                    k = torch.nn.functional.normalize(k, p=2, dim=-1, eps=1e-12)
                attn_logit = self.attention_logit_compute(q, k, beta, att_type='rbf', rbf_l2norm=rbf_l2norm)
                attn = torch.exp(attn_logit)
                if rownorm:
                    attn = row_normalize(attn, rownorm)

            attn_probs = attn.clone() if self.vis else None  # output attentions
            attn_probs_before_calibrate = None
            attn = self.attn_drop(attn)
            # retrieve context from values using attention
            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + v_bg
        else:
            print('Error att_type in SalAttention.')
            exit(0)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_probs, attn_probs_before_calibrate

class SalTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,  # number of transformers
        heads,
        qkv_bias=True,
        qk_same=False,
        mlp_ratio=4.0,  # feedforward networks (MLP)
        attn_dropout=0.0,
        dropout=0.0,
        revised=False,  # the choice of FFN (MLP)
        has_CLS=False,  # whether using CLS token
        vis=False,  # visualize attention maps
        att_type='rbf',  # attention type
        att_pos_enc='',  # it has no effect otherwise att_pos_enc is RPE (relative PE) or CRPE
        grid_size=(12, 12),  # patch grids height and width
    ):
        super(SalTransformer, self).__init__()
        # self.layers = nn.ModuleList([])
        layer_list = []
        self.vis = vis
        self.att_type = att_type

        # assert isinstance(
        #     mlp_ratio, float
        # ), "MLP ratio should be an integer for valid "
        mlp_dim = int(mlp_ratio * dim)

        for _ in range(depth):
            layer_list.append(nn.LayerNorm(dim, eps=1e-6))
            layer_list.append(SalAttention(dim, num_heads=heads, qkv_bias=qkv_bias, qk_same=qk_same, attn_drop=attn_dropout,
                                           proj_drop=dropout, has_CLS=has_CLS, vis=vis, att_type=att_type, att_pos_enc=att_pos_enc, grid_size=grid_size))
            layer_list.append(nn.LayerNorm(dim, eps=1e-6))
            layer_list.append(
                FeedForward(dim, mlp_dim, dropout_rate=dropout) if not revised
                else FeedForward(dim, mlp_dim, dropout_rate=dropout, revised=True)
            )
        self.layers = nn.Sequential(*layer_list)
        self.apply(self._init_weights)  # init LN

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, sal_mask_type='N', sal_mask_list=None, morpho_func=None,
                mask_mode=0, func_mode=0, sal_beta=1.0, sal_eps=0.3, thresh=0.5, use_max=False, rbf_l2norm=False, rownorm=0):
        '''
        x: B x Np x C or B x (1+Np) x C
        sal_mask_list: a list of masks, each of which has size of B x Np
        return a list containing each transformer's feature
        '''

        # for attn, ff in self.layers:
        #     x = attn(x) + x
        #     x = ff(x) + x
        features = []
        attn_probs = []
        vis_morph_masks = []
        morph_reg_loss = []
        for i in range(len(self.layers) // 4):
            # i-th transformer block
            # attn(layer_norm(x)) + x
            layer_id = 4*i
            h = x

            x = self.layers[layer_id](x)  # x=LayerNorm(x)

            # grab saliency mask
            if sal_mask_type == 'N':    # not used
                mask_t = None
                mask_draws = None
            elif sal_mask_type == 'P':  # pure mask
                mask_t = sal_mask_list[0]
                mask_draws = None
            elif sal_mask_type == 'S':  # shared mask; only 1 projection layer + 1 MU
                if i == 0:  # only computed at first ViT
                    mask_t, loss_reg, mask_draws = morpho_func(patch_feats=x, vit_index=0)
                    morph_reg_loss.append(loss_reg)
                    B, _, h, w = mask_t.shape
                    mask_t = mask_t.reshape(B, h * w)  # B x Np; Np = h*w
            elif sal_mask_type == 'IS':  # individual mask; individual projection + 1 MU
                mask_t, loss_reg, mask_draws = morpho_func(patch_feats=x, vit_index=i)
                morph_reg_loss.append(loss_reg)
                B, _, h, w = mask_t.shape
                mask_t = mask_t.reshape(B, h * w)  # B x Np; Np = h*w

            elif sal_mask_type == 'II':  # individual mask; individual projection + individual MU
                mask_t, loss_reg, mask_draws = morpho_func(patch_feats=x, vit_index=i)
                morph_reg_loss.append(loss_reg)
                B, _, h, w = mask_t.shape
                mask_t = mask_t.reshape(B, h * w)  # B x Np; Np = h*w

            x, attn_per_layer, attn_before_calibrate = self.layers[layer_id+1](x,
                                                        sal_mask=mask_t,
                                                        mask_mode=mask_mode,
                                                        func_mode=func_mode,
                                                        beta=sal_beta,
                                                        eps=sal_eps,
                                                        thresh=thresh,
                                                        use_max=use_max,
                                                        rbf_l2norm=rbf_l2norm,
                                                        rownorm=rownorm
                                                        )
            x = x + h  # skip connection

            # ffn(layer_norm(x)) + x
            h = x
            x = self.layers[layer_id+3](self.layers[layer_id+2](x))
            x = x + h  # skip connection

            features.append(x)
            if self.vis:
                attn_probs.append(attn_per_layer)  # 2L x B x N_{heads} x N_{patches} x N_{patches} or None
                attn_probs.append(attn_before_calibrate)
                vis_morph_masks.append(mask_draws) # L x B x 3 x H x W

        return features, attn_probs, vis_morph_masks, morph_reg_loss

    def draw_attn_probs(self, attn_probs, patch_ind, patch_size=(12, 12), save_path_wo_ext=None, save_root='./episode_images/vit_attn_maps'):
        '''
        show specific patch's mean attn_prob (H x W)
        attn_probs: Nh x Np x Np, cuda tensor
        patch_ind:
        patch_size: (H x W), note that H x W = Np
        '''
        import matplotlib.pyplot as plt
        import os
        # save_root = './episode_images/vit_attn_maps'
        if os.path.exists(save_root) is False:
            os.makedirs(save_root)

        attn_probs_t = attn_probs.clone().mean(dim=0)  # Np x Np
        attn_probs_t2 = attn_probs_t[patch_ind].reshape(patch_size).cpu().numpy()
        plt.imshow(attn_probs_t2)  # here we can use different cmap to make it colorful
        if save_path_wo_ext is not None:
            save_path = os.path.join(save_root, save_path_wo_ext + '_' + str(patch_ind) + '.png')
            plt.savefig(save_path, bbox_inches='tight')
        # if does_show:
        #     plt.show()
        attn_probs_t3 = attn_probs[:, patch_ind, :].clone().reshape(-1, *patch_size).cpu().numpy()  # Nh x H x W
        vmax = torch.max(attn_probs[:, patch_ind, :]).item()
        for i in range(attn_probs_t3.shape[0]):  # save Nh attn maps for patch_ind
            plt.imshow(attn_probs_t3[i]/vmax, vmin=0, vmax=1.0)
            save_path = os.path.join(save_root, save_path_wo_ext + '_' + str(patch_ind) + '_h{}'.format(i) + '.png')
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def draw_attn_probs2(self, attn_probs, save_path_wo_ext=None, does_show=False, save_root = './episode_images/vit_attn_maps'):
        '''
        show whole Np x Np attention
        attn_probs: B x Nh x Np x Np, cuda tensor
        '''
        import matplotlib.pyplot as plt
        import os
        # save_root = './episode_images/vit_attn_maps'
        if os.path.exists(save_root) is False:
            os.makedirs(save_root)

        attn_probs_t = attn_probs.clone().mean(dim=1).cpu().numpy() # B x Np x Np
        B = attn_probs_t.shape[0]
        for i in range(B):
            plt.imshow(attn_probs_t[i])
            if save_path_wo_ext is not None:
                save_path = os.path.join(save_root, save_path_wo_ext + '_' + str(i) + '.png')
                plt.savefig(save_path, bbox_inches='tight')
            if does_show:
                plt.show()
            plt.close()




