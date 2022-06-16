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
    def __init__(self, in_chans, num_classes,
                 block_depth=6, dim=128, num_heads = 24,
                 num_it=1,
                 drop_path_rate=0., layer_scale_init_value=0, stem_stride=1
                 ):
        super().__init__()

        self.num_it = num_it
        self.feature_embedder = ConvNeXtFeatures(in_chans=in_chans, depth=block_depth, dim=dim, drop_path_rate=drop_path_rate,
                                                 layer_scale_init_value=layer_scale_init_value, stem_stride=stem_stride)

        self.tograph = ImageToGraphTransformer2(dim, dim, num_heads)

        self.graph = GeometricGraphNN(in_channels=dim,
                          hidden_channels=dim,
                          output_channels=dim,
                          num_layers=block_depth,
                          norm="batch", droprate=0.2, pool='sum', task='graph', conv_depth=1, cond_method='strong',
                          cond_depth=1, use_x_i=False, embedding='mlp', sigma=0.2)

        self.classifier = nn.Sequential(nn.Linear(dim, dim),
                      torch.nn.SiLU(inplace=True),
                      nn.Linear(dim, num_classes))



    def forward(self, x):

        predictions = []
        poss = []

        # Base pixel embedding
        b = self.feature_embedder(x)  # [B, C, X, Y]

        # Scene descriptor
        z = torch.mean(b, dim=(-2,-1))  # [B, C]

        # Prediction from scene descriptor
        predictions.append(self.classifier(z))

        pos = None
        for it in range(self.num_it):
            # Determine capsules
            x, pos = self.tograph(b, z, pos)  # [B, N, C], [B, N, 2]
            # And their pair-wise geometric relations
            edge_attr = pos[:, :, None, :] - pos[:, None, :, :]
            # Update scene representation via graph-based routing
            z = z + self.graph(x, edge_attr)  # [B, N, C]
            # z = self.graph(x, edge_attr)  # [B, N, C]
            predictions.append(self.classifier(z))
            poss.append(pos)

        return predictions[-1], predictions, poss



class GeometricGraphNN(nn.Module):
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
        act_fn = torch.nn.SiLU(inplace=True)

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
                MyConvBlockNeXt(hidden_channels, hidden_channels, edge_embedding_dim, hidden_features=hidden_channels,
                          layers=conv_depth, act_fn=torch.nn.SiLU(inplace=True), cond_method=cond_method, cond_depth=cond_depth,
                          use_x_i=use_x_i, aggr="mean", norm=norm, droprate=droprate))
        self.layers = nn.ModuleList(layers)

        self.pre_pool = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                      act_fn,
                                      nn.Linear(hidden_channels, hidden_channels))
        self.post_pool = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                       act_fn,
                                       nn.Linear(hidden_channels, output_channels))

    def forward(self, x, edge_attr):

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
            x = layer(x, edge_embedded)  # [B, N, C]

        x = self.pre_pool(x)
        x = torch.mean(x, dim=-2)  # [B, C]
        x = self.post_pool(x)

        # Return result
        return x

class MyConvBlockNeXt(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 domain_features,
                 hidden_features=128,
                 layers=1,
                 act_fn=torch.nn.SiLU(inplace=True),
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
        self.conv_layer_1 = MyConv(in_features, hidden_features, domain_features, hidden_features, layers, act_fn,
                                 cond_method, cond_depth, use_x_i, aggr)
        self.linear_1 = torch.nn.Linear(hidden_features, 4 * hidden_features)
        self.linear_2 = torch.nn.Linear(4 * hidden_features, out_features)
        self.layer_scale = torch.nn.Parameter(torch.ones(1, out_features) * layer_scale)

        if norm == "batch":
            self.norm = LayerNorm(hidden_features, eps=1e-6, data_format="channels_last")
        else:
            self.norm = torch.nn.Identity()


        self.equalInOut = (in_features == out_features)
        self.skip_connect = torch.nn.Identity() if self.equalInOut else torch.nn.Linear(in_features, out_features)

        self.droprate = droprate

    def forward(self, x, edge_attr):
        out = x  # [B, N, C]
        if self.droprate > 0.:
            out = torch.nn.functional.dropout(out, p=self.droprate, training=self.training)
        out = self.conv_layer_1(out, edge_attr)
        out = self.norm(out)
        out = self.linear_1(out)
        out = self.act_fn(out)
        out = self.linear_2(out)
        out = self.layer_scale * out
        out = out + self.skip_connect(x)
        return out

class MyConv(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 domain_features,
                 hidden_features=128,
                 layers=1,
                 act_fn=torch.nn.SiLU(inplace=True),
                 cond_method='strong',
                 cond_depth=1,
                 use_x_i=False,
                 aggr="mean"):
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

    def forward(self, x, edge_attr):
        [B, N, C] = x.shape
        if self.use_x_i:
            m = torch.cat((x[:, :, None, :].expand(B, N, N, C), x[:, None, :, :].expand(B, N, N, C)), dim=-1)
        else:
            m = x[:, None, :, :].expand(B, N, N, C)

        for i in range(self.layers):
            # Get the layer
            layer = self.message_layers[i]
            # Apply the layer
            if isinstance(layer, ConditionalLinear):
                m = layer(m, edge_attr)
            elif isinstance(layer, nn.Linear):
                m = layer(m)
            # Do not apply activation function for the output layer
            if i != self.layers - 1:
                m = self.act_fn(m)
        x = torch.mean(m, dim=-2)  # [B, N, N, C] -> [B, N, C]
        return x




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
        self.rel_pos_proj = nn.Linear(2, int(embed_dim / num_heads))

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)

    def forward(self, x, z, pos=None):  # x.shape = [B, C, X, Y]
        x_flat = torch.flatten(x, -2, -1)  # [B, C, XY]
        x_flat = x_flat.transpose(-1, -2)  # [B, XY, C]
        pos_grid = get_coords(x.shape[-2], x.shape[-1]).type_as(x)  # [XY, 2]
        batch_size, seq_length, embed_dim = x_flat.shape

        q = self.q_proj(z).reshape(batch_size, self.num_heads, self.head_dim)  # [B, H, C']
        k = self.k_proj(x_flat).reshape(batch_size, seq_length, self.num_heads, self.head_dim)  # [B, XY, H, C']
        if pos is not None:  # [None, XY, None, 2] - [B, None, H, 2]
            rel_pos_grid = pos_grid[None, :, None, :] - pos[:, None, :, :]  # [B, XY, H, 2]
            qxk = self.rel_pos_proj(rel_pos_grid)  # [B, XY, H, C']
            k = k * qxk
        v = self.v_proj(x_flat)  # [B, XY, C]

        # Determine value outputs
        d_k = q.size()[-1]

        attn_logits = torch.einsum('bhc,bnhc->bnh', q, k)  # [B, XY, H]
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-2)  # [B, XY, H]

        # Weighted average pos (per head)
        pos = torch.einsum('bnh,nx->bhx', attention, pos_grid)  # [B, H, 2]
        # Weighted average value (per head)
        values = torch.einsum('bnh,bnc->bhc', attention, v)  # [B, H, C]

        # To graph object
        return values, pos  # [B, N, C], [B, N, 2]


def get_coords(h, w):
    # return a coordinate grid over [0, 1] interval with h (heigh) and w (width) sample density
    range_x = torch.tensor(np.linspace(0, h - 1, h))
    range_y = torch.tensor(np.linspace(0, w - 1, w))

    xx, yy = torch.meshgrid((range_x, range_y), indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)





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
            std = torch.std(out, dim=(0,1,2))
            self.net[-1].weight.data /= std[:, None]





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

        self.to_pre_q = nn.Linear(dim // num_slots, dim)
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

    def forward(self, inputs, z, num_slots=None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        # mu = self.slots_mu.expand(b, n_s, -1)
        mu = self.to_pre_q(z.reshape(b,n_s,-1))
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


class ImageToGraphTransformer2(nn.Module):

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

    def forward(self, x, z, pos=None):  # x.shape = [B, C, X, Y]
        x_flat = torch.flatten(x, -2, -1)  # [B, C, XY]
        x_flat = x_flat.transpose(-1, -2)  # [B, XY, C]
        pos_grid = get_coords(x.shape[-2], x.shape[-1]).type_as(x)  # [XY, 2]

        values, attn = self.slot_attn(x_flat, z)  # shapes: [B, H, C], [B, H, XY]
        pos = torch.einsum('bhn,nx->bhx', attn, pos_grid)  # [B, H, 2]

        # To graph object
        return values, pos  # [B, N, C], [B, N, 2]