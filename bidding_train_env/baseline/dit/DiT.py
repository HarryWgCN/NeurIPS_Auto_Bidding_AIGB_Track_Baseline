# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import os
from torch.optim import Adam

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import  Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class XEmbedder(nn.Module):
    """
    Embeds scalar x into vector representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(17, hidden_size, bias=True),#TODO: fix 17
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    def forward(self, x):
        print("x.shape",x.dtype,)
        print(self.mlp[0].weight.dtype)
        x=x.to(self.mlp[0].weight.dtype)
        print("x.shape",x.dtype,)
        x_emb = self.mlp(x)
        return x_emb

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        print("x.shape,c.shpae",x.shape,c.shape)
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        print(" shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp", shift_msa.shape, scale_msa.shape, gate_msa.shape, shift_mlp.shape, scale_mlp.shape, gate_mlp.shape)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size,action_dim, hidden_dim_final_layer=None, ):
        super().__init__()
        if hidden_dim_final_layer is None:
            hidden_dim_final_layer= hidden_size // 2
        self.layer =  nn.Sequential(
            nn.Linear(hidden_size, hidden_dim_final_layer),
            nn.ReLU(),
            nn.Linear(hidden_dim_final_layer, hidden_dim_final_layer),
            nn.ReLU(),
            nn.Linear(hidden_dim_final_layer, hidden_dim_final_layer),
            nn.ReLU(),
            nn.Linear(hidden_dim_final_layer, action_dim),
        )

    def forward(self, x,c):
        x = self.layer(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        # input_size=32,
        # patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        dim_actions=1,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        # self.patch_size = patch_size
        self.num_heads = num_heads
        self.action_dim = dim_actions

        self.x_embedder = XEmbedder( hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # TODO: get2d->get1d:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 1)
        # pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.x_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.x_embedder.mlp[2].weight, std=0.02)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero-out output layers:
        def init_linear_weights(m):
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化权重
                nn.init.xavier_uniform_(m.weight)
                # 使用常数初始化偏置
                nn.init.constant_(m.bias, 0)
        self.final_layer.layer.apply(init_linear_weights)
        # nn.init.constant_(self.final_layer.layer, 0)

    # def unpatchify(self, x):
    #     """
    #     x: (N, T, patch_size**2 * C)
    #     imgs: (N, H, W, C)
    #     """
    #     c = self.out_channels
    #     p = self.x_embedder.patch_size[0]
    #     h = w = int(x.shape[1] ** 0.5)
    #     assert h * w == x.shape[1]
    #
    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    #     return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        print("dit forward",x.shape,t.shape,y.shape)# dit forward torch.Size([1000, 48, 17]) torch.Size([1000]) torch.Size([1000])
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        print("dit forward2",x.shape,t.shape,y.shape)# dit forward2 torch.Size([1000, 48, 384]) torch.Size([1000, 384]) torch.Size([1000, 384])
        for block in self.blocks:
            print(x.shape,c.shape,"xtyc",t.shape,y.shape)# torch.Size([1, 48, 384]) torch.Size([48, 384])
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)       # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        # x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    # def forward(self, x, t, y):
    #     """
    #     Forward pass of DiT.
    #     x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
    #     t: (N,) tensor of diffusion timesteps
    #     y: (N,) tensor of class labels
    #     """
    #     x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
    #     t = self.t_embedder(t)                   # (N, D)
    #     y = self.y_embedder(y, self.training)    # (N, D)
    #     c = t + y                                # (N, D)
    #     for block in self.blocks:
    #         x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)       # (N, T, D)
    #     x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
    #     x = self.unpatchify(x)                   # (N, out_channels, H, W)
    #     return x
    #
    # def forward_with_cfg(self, x, t, y, cfg_scale):
    #     """
    #     Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
    #     """
    #     # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    #     half = x[: len(x) // 2]
    #     combined = torch.cat([half, half], dim=0)
    #     model_out = self.forward(combined, t, y)
    #     # For exact reproducibility reasons, we apply classifier-free guidance on only
    #     # three channels by default. The standard approach to cfg applies it to all channels.
    #     # This can be done by uncommenting the following line and commenting-out the line following that.
    #     # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    #     eps, rest = model_out[:, :3], model_out[:, 3:]
    #     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    #     half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    #     eps = torch.cat([half_eps, half_eps], dim=0)
    #     return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024,  num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768,  num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768,  num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384,  num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384,  num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


class DIT(nn.Module):
    def __init__(self, dim_obs=16, dim_actions=1, gamma=1, tau=0.01, lr=1e-4,
                 network_random_seed=200,
                 ACTION_MAX=10, ACTION_MIN=0,
                 step_len=48, n_timesteps=10,model_name='DiT-S/2'):

        super().__init__()

        self.n_timestamps = n_timesteps
        self.num_of_states = dim_obs
        self.num_of_actions = dim_actions
        self.ACTION_MAX = ACTION_MAX
        self.ACTION_MIN = ACTION_MIN
        self.network_random_seed = network_random_seed
        self.step_len = step_len
        self.model = DiT_models[model_name](
            # input_size=32,
            in_channels=4,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=True,
        )
        # Note that parameter initialization is done within the DiT constructor
        # self.model = self.model.to(device)

        self.step = 0

        torch.random.manual_seed(network_random_seed)

        self.num_of_episodes = 0

        self.GAMMA = gamma
        self.tau = tau
        self.num_of_steps = 0
        # cuda usage
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()

        self.diffuser_lr = lr

        self.diffuserModel_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.invModel_optimizer = torch.optim.Adam(self.model.inv_model.parameters(), lr=lr)

    def toCuda(self):
        self.model.cuda()

    def trainStep(self, states, actions, returns, masks,diffusion):
        self.model.train()
        if self.use_cuda:
            self.model.cuda()
            states = states.cuda()
            actions = actions.cuda()
            returns = returns.cuda()
            masks = masks.cuda()

        x = torch.cat([actions, states], dim=-1)
        # cond = torch.ones_like(states[:, 0], device=states.device)[:, None, :]
        # loss, infos, (diffuse_loss, inv_loss) = self.model.loss(x, cond, returns=returns, masks=masks)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=states.device)
        y = torch.zeros_like(t)
        model_kwargs = dict(y=y)
        loss_dict = diffusion.training_losses(self.model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        loss.backward()
        self.diffuserModel_optimizer.step()
        self.diffuserModel_optimizer.zero_grad()

        return loss

    def forward(self, x: torch.Tensor):
        if len(list(x.shape)) < 2:
            x = torch.reshape(x, [48, self.num_of_states + 1])
        else:
            x = x[0][0]
        cur_time = int(x[0][-1].item())
        cur_time = cur_time + 1
        states = x[:cur_time]
        states = states[:, :-1]
        conditions = states
        returns = torch.tensor([[1.0]], device=x.device)
        #TODO: >t y
        t = torch.Tensor([cur_time]).int().to(x.device)
        y = torch.zeros_like(t).int().to(x.device)
        print(x.shape,t.shape,"!!!")#torch.Size([48, 17]) torch.Size([48, 1]) !!!
        x_0 = self.model(x=x,t=t,y=y)

        states = x_0[0, :cur_time + 1]
        states_next = states[None, -1]
        if cur_time > 1:
            states_curt1 = conditions[-2].float()[None, :]
        else:
            states_curt1 = torch.zeros_like(states_next, device=states_next.device)
        if cur_time > 2:
            states_curt2 = conditions[-3].float()[None, :]
        else:
            states_curt2 = torch.zeros_like(states_next, device=states_next.device)
        states_comb = torch.hstack([states_curt1, states_curt2, conditions[-1].float()[None, :], states_next])
        # actions = self.model.inv_model(states_comb)
        #TODO:>
        t = torch.Tensor([cur_time]).int().to(x.device)
        y = torch.zeros_like(t).int().to(x.device)
        actions = self.model(states_comb,t=t,y=y)
        actions = actions.detach().cpu()[0]  # .cpu().data.numpy()
        return actions

    def save_net(self, save_path, epi):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # checkpoint = {
        #     "model": self.model.module.state_dict(),
        #     "ema": self.ema.state_dict(),
        # }
        # torch.save(checkpoint, f'{save_path}/dit.pt')
        torch.save(self.model.state_dict(), f'{save_path}/dit.pt')

    def save_model(self, save_path, epi):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        model_temp = self.cpu()
        jit_model = torch.jit.script(model_temp)
        torch.jit.save(jit_model, f'{save_path}/dit_{epi}.pth')

    def load_net(self, load_path="saved_model/fixed_initial_budget_dit", device='cuda:0'):
        self.model.load_state_dict(torch.load(load_path, map_location='cpu'))
        self.optimizer = Adam(self.model.parameters(), lr=self.diffuser_lr)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()
