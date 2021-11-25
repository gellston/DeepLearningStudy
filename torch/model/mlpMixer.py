import torch
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce



class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super(MixerBlock, self).__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            # (n_samples, n_patches, dim(channel)) -> (n_samples, dim(channel), n_patches)
            MLP(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d'),
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MLP(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super(MLPMixer, self).__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size'
        self.num_patch = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),  # h,w = num_patch ** 0.5
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(
                MixerBlock(dim, self.num_patch, token_dim, channel_dim)
            )

        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)  # (2, 196, 512)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)  # (2, 196, 512)
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # global average pooling (2, 512)  / sequential을 사용했다면 Reduce('b n d -> b d', 'mean') 도 가능
        x = self.mlp_head(x)  # (2, 1000)
        return x