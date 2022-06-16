# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from .convnext import Block, LayerNorm
import math
import numpy as np

import torch_geometric
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import LayerNorm as LayerNorm2




# For plot
import io
import PIL
import matplotlib.pyplot as plt
import wandb
import torchvision





class ConvNeXtFeatures(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3,
                 depth=18, dim=384, drop_path_rate=0.,
                 layer_scale_init_value=0, stem_stride=1
                 ):
        super().__init__()

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=7, stride=stem_stride, padding=3)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i],
                                    layer_scale_init_value=layer_scale_init_value)
                                    for i in range(depth)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class GrapsuleNet(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3,
                 depth=18, dim=384, drop_path_rate=0.,
                 layer_scale_init_value=0, stem_stride=1
                 ):
        super().__init__()

        self.feature_embedder = ConvNeXtFeatures(in_chans=in_chans, depth=6, dim=dim, drop_path_rate=drop_path_rate,
                                                 layer_scale_init_value=layer_scale_init_value, stem_stride=stem_stride)

        self.tograph = ImageToGraphTransformer(dim, dim, 24)

        self.graph = PONI(in_channels=dim,
                          hidden_channels=128,
                          output_channels=10,
                          num_layers=6,
                          norm="batch",
                          droprate=0.2,  # 0.2
                          pool='sum',
                          task='graph',
                          conv_depth=1,
                          cond_method='strong',
                          cond_depth=1,
                          use_x_i=False,
                          embedding='mlp',
                          sigma=0.2)

        self.im_to_log = None
        self.im_to_log2 = None
        self.penalty = None

    def forward(self, x):
        # Plot log (part 1)
        imstack2show = (x - x.min()) / (x.max() - x.min())
        [B, C, X, Y] = imstack2show.shape

        x = self.feature_embedder(x)
        x, pos, batch, att, penalty = self.tograph(x)
        # edge_index = knn_graph(pos, k=5, batch=batch, loop=True)
        edge_index = radius_graph(pos, r=100., loop=True)
        edge_attr = pos[edge_index[0]] - pos[edge_index[1]]  # Message comes from this direction
        batch = batch.type_as(edge_index)

        # print(torch.std(pos))

        # Plot log (part 2)
        imnr = 0
        buf = io.BytesIO()
        plt.figure()
        fig = plt.imshow(imstack2show[imnr].permute([1, 2, 0]).cpu().detach().numpy())
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.scatter(pos[batch==imnr, 1].cpu().detach().numpy(), pos[batch==imnr, 0].cpu().detach().numpy(), s=100, c=np.linspace(0, 1, 24), alpha=0.8, cmap='jet')
        plt.scatter(pos[batch == imnr, 1][:1].cpu().detach().numpy(), pos[batch == imnr, 0][:1].cpu().detach().numpy(), s=700,
                    c=np.linspace(0, 1, 24)[:1], alpha=0.8, cmap='jet')
        plt.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)
        plt.close()
        self.im_to_log = wandb.Image(img)
        # end Plot log

        # Plot part (3)
        heatmaps = att[imnr, :].permute(1,0)
        heatmaps = heatmaps/torch.max(heatmaps,-1,keepdim=True)[0]
        heatmaps = heatmaps.reshape(24,1,X,Y)
        image_grid = torchvision.utils.make_grid(heatmaps, 4, 0).permute([1, 2, 0])
        buf = io.BytesIO()
        plt.figure()
        fig = plt.imshow(image_grid.cpu().detach().numpy())
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)
        plt.close()
        self.im_to_log2 = wandb.Image(img)
        # end plot part 3
        self.penalty = penalty

        x = self.graph(x, batch, edge_attr, edge_index)

        return x


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class ImageToGraphTransformer(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)

    def forward(self, x):
        x_flat = torch.flatten(x, -2, -1)  # [B, C, XY]
        x_flat = x_flat.transpose(-1, -2)  # [B, XY, C]
        x_mean = torch.mean(x_flat, -2)  # [B, C]

        batch_size, seq_length, embed_dim = x_flat.shape

        q = self.q_proj(x_mean).reshape(batch_size, self.num_heads, self.head_dim)  # [B, H, C']
        k = self.k_proj(x_flat).reshape(batch_size, seq_length, self.num_heads, self.head_dim)  # [B, XY, H, C']
        v = self.v_proj(x_flat)  # [B, XY, C]

        # Determine value outputs
        d_k = q.size()[-1]

        attn_logits = torch.einsum('bhc,bnhc->bnh', q, k)  # [B, XY, H]
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-2)  # [B, XY, H]

        # Weighted pooling
        pos_grid = get_coords(x.shape[-2],x.shape[-1]).type_as(x)  # [XY, 2]
        values = torch.einsum('bnh,bnc->bhc', attention, v)  # [B, H, C]
        pos = torch.einsum('bnh,nx->bhx', attention, pos_grid)  # [B, H, 2]
        batch = torch.arange(0,batch_size).repeat_interleave(self.num_heads).type_as(x)  # [B*H]

        # pos_std
        target_sigma = 3
        # [-1,XY,2,-1] * [B, XY, -1, H]
        mus = torch.sum(pos_grid[None,:,:, None] * attention[:, :, None, :], dim=1)  # [B, 2, H]
        # [B, XY, None, H] ([-1, XY, 2, -1] - [B, None, 2, H])
        variance = torch.sum(attention[:, :, None, :] * (pos_grid[None, :, :, None] - mus[:, None, :, :])**2, dim=1)  # [B, 2, H]
        penalty = torch.mean(torch.abs(variance - target_sigma**2 * torch.ones_like(variance)))

        # [B,XY,H] * [B, XY, C]
        # [B, XY, H, C] * [B, XY, H, C]
        # variance = torch.sum(attention[:, :, :, None] * (v[:, :, None, :] - values[:, None, :, :]) ** 2,
        #                      dim=1)  # [B, H, C]
        # penalty = torch.mean(variance)

        values = values.view(batch_size * self.num_heads, -1)
        pos = pos.view(batch_size * self.num_heads, 2)

        penalty = 0.

        # To graph object
        return values, pos, batch, attention, penalty


class ImageToGraphTransformer2(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.x_predict = nn.Sequential(nn.Linear(input_dim, embed_dim),
                                             torch.nn.SiLU(),
                                             nn.Linear(embed_dim, num_heads * 2),
                                             torch.nn.Sigmoid())
        self.v_proj = nn.Linear(input_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        self.x_predict[-2].weight.data *= 2.


    def forward(self, x):
        x_flat = torch.flatten(x, -2, -1)  # [B, C, XY]
        x_flat = x_flat.transpose(-1, -2)  # [B, XY, C]
        x_mean = torch.mean(x_flat, -2)  # [B, C]

        batch_size, seq_length, embed_dim = x_flat.shape

        v = self.v_proj(x_flat) # [B, XY, C]
        pos = 31 * self.x_predict(x_mean).reshape(batch_size, self.num_heads, 2)  # [B, H, 2]

        pos_grid = get_coords(x.shape[-2], x.shape[-1]).type_as(x)  # [XY, 2]
        attention = sample_gaussians(pos_grid, pos, 3)  # [B, XY, H]

        # Weighted pooling
        values = torch.einsum('bnh,bnc->bhc', attention, v).view(batch_size * self.num_heads, -1)  # [B, H, C]
        batch = torch.arange(0,batch_size).repeat_interleave(self.num_heads).type_as(x)  # [B*H]

        # pos_std
        penalty = torch.sum(attention, -2)
        penalty = torch.mean((penalty - torch.ones_like(penalty))**2)

        pos = pos.view(batch_size * self.num_heads, 2)

        # To graph object
        return values, pos, batch, attention, penalty


class ImageToGraphTransformer3(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)

    def forward(self, x):
        x_flat = torch.flatten(x, -2, -1)  # [B, C, XY]
        x_flat = x_flat.transpose(-1, -2)  # [B, XY, C]
        x_mean = torch.mean(x_flat, -2)  # [B, C]

        batch_size, seq_length, embed_dim = x_flat.shape

        q = self.q_proj(x_mean).reshape(batch_size, self.num_heads, self.head_dim)  # [B, H, C']
        k = self.k_proj(x_flat).reshape(batch_size, seq_length, self.num_heads, self.head_dim)  # [B, XY, H, C']
        v = self.v_proj(x_flat)  # [B, XY, C]

        # Determine value outputs
        d_k = q.size()[-1]

        attn_logits = torch.einsum('bhc,bnhc->bnh', q, k)  # [B, XY, H]
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-2)  # [B, XY, H]

        # Weighted pooling
        pos_grid = get_coords(x.shape[-2],x.shape[-1]).type_as(x)  # [XY, 2]
        values = torch.einsum('bnh,bnc->bhc', attention, v).view(batch_size * self.num_heads, -1)  # [B, H, C]
        pos = torch.einsum('bnh,nx->bhx', attention, pos_grid)  # [B, H, 2]
        batch = torch.arange(0,batch_size).repeat_interleave(self.num_heads).type_as(x)  # [B*H]

        # pos_std
        # desired_attention = sample_gaussians(pos_grid, pos, 3)  # [B, XY, H]
        desired_attention = F.softmax(sample_log_gaussians(pos_grid, pos, 3),dim=-2)
        penalty = torch.nn.functional.cross_entropy(attn_logits, desired_attention)

        pos = pos.view(batch_size * self.num_heads, 2)

        # To graph object
        return values, pos, batch, attention, penalty


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        torch.nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn

class ImageToGraphTransformer4(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.slot_attn = SlotAttention(
            num_slots=num_heads,
            dim=embed_dim,
            iters=3  # iterations of attention, defaults to 3
        )

    def forward(self, x):
        x_flat = torch.flatten(x, -2, -1)  # [B, C, XY]
        x_flat = x_flat.transpose(-1, -2)  # [B, XY, C]

        batch_size, seq_length, embed_dim = x_flat.shape

        # Weighted pooling
        pos_grid = get_coords(x.shape[-2],x.shape[-1]).type_as(x)  # [XY, 2]

        values, attn = self.slot_attn(x_flat)  # shapes: [B, H, C], [B, H, XY]
        pos = torch.einsum('bhn,nx->bhx', attn, pos_grid)  # [B, H, 2]
        batch = torch.arange(0,batch_size).repeat_interleave(self.num_heads).type_as(x)  # [B*H]

        # target_sigma = 3
        # # [B, H, XY, None] ([-1, -1, XY, 2] - [B, H, None, 2])
        # variance = torch.sum(attn[:, :, :, None] * (pos_grid[None, None, :, :] - pos[:, :, None, :])**2, dim=2)  # [B, H, 2]
        # penalty = torch.mean(torch.abs(variance - target_sigma**2 * torch.ones_like(variance)))

        penalty = -torch.sum(torch.max(attn, 1)[0])



        values = values.view(batch_size * self.num_heads, -1)
        pos = pos.view(batch_size * self.num_heads, 2)


        # To graph object
        return values, pos, batch, attn.permute(0,2,1), penalty


def sample_gaussians(grid, mus, sigma):
    rel_pos = grid[None, :, None, :] - mus[:,None, :, :]
    sampled = (1/(2*torch.pi*sigma**2)) * torch.exp(-0.5 * torch.sum(rel_pos**2,-1) / (sigma**2))  # [B, XY, H]
    return sampled

def sample_log_gaussians(grid, mus, sigma):
    rel_pos = grid[None, :, None, :] - mus[:,None, :, :]
    sampled = -0.5 * torch.sum(rel_pos**2,-1) / (sigma**2)  # [B, XY, H]
    return sampled

def get_coords(h, w):
    # return a coordinate grid over [0, 1] interval with h (heigh) and w (width) sample density
    range_x = torch.tensor(np.linspace(0, h - 1, h))
    range_y = torch.tensor(np.linspace(0, w - 1, w))

    xx, yy = torch.meshgrid((range_x, range_y), indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention



class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = torch.nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = torch.nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = torch.nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = torch.nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = torch.nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = torch.nn.modules.linear.NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = torch.nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = torch.nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.q_proj_weight)
            torch.nn.init.xavier_uniform_(self.k_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            torch.nn.init.constant_(self.in_proj_bias, 0.)
            torch.nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            torch.nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            torch.nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask = None,
                need_weights = True, attn_mask = None,
                average_attn_weights = True):

        # query.shape = [B, C]
        # key.shape = [B, C, X, Y]
        # value.shape = [B, C, X, Y]

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights




class PONI(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 output_channels,
                 num_layers,
                 norm="batch",
                 droprate=0.0,
                 pool='sum',
                 task='graph',
                 conv_depth=1,
                 cond_method='weak',
                 cond_depth=1,
                 use_x_i=False,
                 embedding='identity',
                 sigma=0.2):
        super().__init__()

        # ############################ Two cases, stay on R3 or lift to R3xS2 ##########################################

        edge_attr_dim = 2
        in_channels = in_channels

        # ################################### Settings for the graph NN ################################################

        self.task = task
        act_fn = torch.nn.SiLU()

        # ################### Settings for pre-embedding the geometric conditioning vector. ############################

        if not embedding == 'identity':
            self.calibrated = False
            edge_embedding_dim = hidden_channels
            self.edge_embedding_fn = RFFNet(edge_attr_dim, edge_embedding_dim, [hidden_channels, hidden_channels], sigma=sigma)
        else:
            self.calibrated = True
            edge_embedding_dim = edge_attr_dim
            self.edge_embedding_fn = torch.nn.Identity()


        # ##################################### The graph NN layers ####################################################

        # Initial node embedding layer
        self.embedding_layer = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                             act_fn,
                                             nn.Linear(hidden_channels, hidden_channels))

        # Message passing layers
        layers = []
        for i in range(num_layers):
            layers.append(
                ConvBlockNeXt(hidden_channels, hidden_channels, edge_embedding_dim, hidden_features=hidden_channels,
                          layers=conv_depth, act_fn=torch.nn.SiLU(), cond_method=cond_method, cond_depth=cond_depth,
                          use_x_i=use_x_i, aggr="mean", norm=norm, droprate=droprate))
        self.layers = nn.ModuleList(layers)

        # Readout layers
        if task == 'graph':
            self.pre_pool = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                          act_fn,
                                          nn.Linear(hidden_channels, hidden_channels))
            self.post_pool = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                           act_fn,
                                           nn.Linear(hidden_channels, output_channels))
            self.init_pooler(pool)
        elif task == 'node':
            self.pre_pool = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                          act_fn,
                                          nn.Linear(hidden_channels, output_channels))


    def init_pooler(self, pool):
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def forward(self, x, batch, edge_attr, edge_index):

        # Embed the conditioning vectors
        if not self.calibrated:
            print('\n Calibrating embedding function!')
            self.edge_embedding_fn.calibrate(edge_attr)
            self.calibrated = True
        edge_embedded = self.edge_embedding_fn(edge_attr)

        # Embed
        x = self.embedding_layer(x)

        # Pass messages
        for layer in self.layers:
            x = layer(x, edge_index, edge_embedded, batch)

        # Pre pool
        x = self.pre_pool(x)

        if self.task == 'graph':
            # Pool over nodes
            x = self.pooler(x, batch)
            # Predict
            x = self.post_pool(x)

        # Return result
        return x





class RFFEmb(nn.Module):
    def __init__(self, in_features, out_features, sigma, trainable):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.trainable = trainable
        B = torch.randn(int(out_features/2), in_features)
        B *= self.sigma*2*math.pi

        self.correction = math.sqrt(2)

        if trainable:
            B = nn.Parameter(B)
        self.register_buffer("B", B, persistent=True)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, sigma={}, trainable={}'.format(
            self.in_features, self.out_features, self.sigma, self.trainable
        )

    def forward(self, x):
        out = F.linear(x, self.B)
        out = torch.cat((out.sin(), out.cos()), dim=-1)
        out = out*self.correction  # Will affect backprop when trainable
        return out


class RFFNet(nn.Module):
    """ RFF net to parameterise convs """

    def __init__(self, in_features, out_features, hidden_features, sigma=1., activation=nn.ReLU, trainable=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation = activation
        self.sigma = sigma

        assert hidden_features[0] % 2 == 0, "Please use an even number of features for the RFF embedding"

        net = []

        dims = [in_features] + hidden_features

        for i in range(len(dims)-1):
            if i == 0:
                net.append(RFFEmb(dims[i], dims[i+1], sigma=sigma, trainable=trainable))
            else:
                net.append(nn.Linear(dims[i], dims[i+1]))
                net.append(activation())

        net.append(nn.Linear(dims[-1], out_features))

        self.net = nn.Sequential(*net)
        self.init()

    def init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                if layer == self.net[-1]:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                else:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        out = self.net(x)
        return out

    def calibrate(self, x):
        with torch.no_grad():
            out = self.forward(x)
            std = torch.std(out, dim=0)
            print(std)
            self.net[-1].weight.data /= std[:, None]





class Conv(torch_geometric.nn.MessagePassing):
    def __init__(self,
                 in_features,
                 out_features,
                 domain_features,
                 hidden_features=128,
                 layers=1,
                 act_fn=torch.nn.SiLU(),
                 cond_method='strong',
                 cond_depth=1,
                 use_x_i=False,
                 aggr="mean"):  # TODO: add option basis_fn
        super(Conv, self).__init__(node_dim=0, aggr=aggr)

        # Dimensionalities of the feature vectors
        self.in_features = in_features
        self.out_features = out_features
        self.domain_features = domain_features
        self.hidden_features = hidden_features  # For non-linear conv
        self.layers = layers
        # Layer specs
        self.act_fn = act_fn
        # Conditioning method
        self.cond_method = cond_method
        self.cond_depth = cond_depth
        self.use_x_i = use_x_i

        # Message network layers
        message_layers = []
        in_channels = 2 * in_features if use_x_i else in_features
        for i in range(layers):
            out_channels = out_features if (i == layers - 1) else hidden_features
            if (i < self.cond_depth) and (domain_features != 0):
                layer = ConditionalLinear(in_channels, out_channels, domain_features, method=cond_method)
            else:
                layer = nn.Linear(in_channels, out_channels)
            in_channels = hidden_features
            message_layers.append(layer)
        self.message_layers = nn.ModuleList(message_layers)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        out = torch.cat((x_i, x_j), dim=-1) if self.use_x_i else x_j
        for i in range(self.layers):
            # Get the layer
            layer = self.message_layers[i]
            # Apply the layer
            if isinstance(layer, ConditionalLinear):
                out = layer(out, edge_attr)
            elif isinstance(layer, nn.Linear):
                out = layer(out)
            # Do not apply activation function for the output layer
            if i != self.layers - 1:
                out = self.act_fn(out)
        # Return the result
        return out

    def update(self, message_aggr):
        return message_aggr


class ConvBlockNeXt(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 domain_features,
                 hidden_features=128,
                 layers=1,
                 act_fn=torch.nn.SiLU(),
                 cond_method='strong',
                 cond_depth=1,
                 use_x_i=False,
                 aggr="mean",
                 norm=None,
                 droprate=0.0,
                 layer_scale=1e-6):  # TODO: add option basis_fn
        super().__init__()

        # Dimensionalities of the feature vectors
        self.in_features = in_features
        self.out_features = out_features
        self.domain_features = domain_features
        self.hidden_features = hidden_features  # For non-linear conv
        self.layers = layers
        # Layer specs
        self.act_fn = act_fn
        # Conditioning method
        self.cond_method = cond_method
        self.cond_depth = cond_depth
        self.use_x_i = use_x_i

        # Message network layers
        self.conv_layer_1 = Conv(in_features, hidden_features, domain_features, hidden_features, layers, act_fn,
                                 cond_method, cond_depth, use_x_i, aggr)
        self.linear_1 = torch.nn.Linear(hidden_features, 4 * hidden_features)
        self.linear_2 = torch.nn.Linear(4 * hidden_features, out_features)
        self.layer_scale = torch.nn.Parameter(torch.ones(1, out_features) * layer_scale)

        # with torch.no_grad():
        #     self.conv_layer_1.message_layers[0].linear.weight *= np.sqrt(2.)
        #     self.conv_layer_1.message_layers[0].linear_embedding.weight *= np.sqrt(2.)
        #     self.linear_1.weight *= np.sqrt(2.)
        #     self.linear_2.weight *= np.sqrt(2.)

        if norm == "batch":
            # self.norm = BatchNorm(hidden_features)
            self.norm = LayerNorm2(hidden_features)
        else:
            self.norm = torch.nn.Identity()


        self.equalInOut = (in_features == out_features)
        self.skip_connect = torch.nn.Identity() if self.equalInOut else torch.nn.Linear(in_features, out_features)

        self.droprate = droprate

    def forward(self, x, edge_index, edge_attr, batch=None):
        out = x
        if self.droprate > 0.:
            out = torch.nn.functional.dropout(out, p=self.droprate, training=self.training)
        out = self.conv_layer_1(out, edge_index, edge_attr)
        out = self.norm(out, batch)
        out = self.linear_1(out)
        out = self.act_fn(out)
        out = self.linear_2(out)
        out = self.layer_scale * out
        out = out + self.skip_connect(x)
        return out





class ConditionalLinear(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 cond_features,
                 method='weak'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cond_features = cond_features
        self.method = method

        # Using torch nn Linear and BiLinear
        # The weights that parametrize the conditional linear layer
        if method == 'weak':
            self.linear_weak = torch.nn.Linear(self.in_features + self.cond_features, self.out_features)
        elif method == 'strong':
            self.linear = torch.nn.Linear(self.in_features, self.out_features)
            self.linear_embedding = torch.nn.Linear(self.cond_features, self.out_features, bias=False)
        elif method == 'pure':
            self.bilinear = torch.nn.Bilinear(self.in_features, self.cond_features, self.out_features)
        else:
            raise ValueError('Unknown method, should be \'weak\', \'strong\', or \'pure\'.')

    def forward(self, f_in, cond_vec):
        if self.method == 'weak':
            f_out = self.linear_weak(torch.cat((f_in, cond_vec), dim=-1))
        elif self.method == 'strong':
            f_out = self.linear(f_in) * self.linear_embedding(cond_vec)
        elif self.method == 'pure':
            f_out = self.bilinear(f_in, cond_vec)
        return f_out
