import torch
import einops
import einops.layers.torch
from utils.config import CONFIG

import torch.nn as nn

from modules.transformer import Attention, PreNorm, FeedForward, Transformer

MODEL='model2'

class ViViT(nn.Module):

    def __init__(self,
            image_size,
            patch_size,
            num_frames,
            dim = 192,
            depth = 4,
            heads = 3,
            pool = 'cls',
            in_channels = 3,
            dim_head = 64,
            dropout = 0.,
            emb_dropout = 0.,
            scale_dim = 4):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            einops.layers.torch.Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.parameter.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.parameter.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.parameter.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_frames)
        )


    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = einops.repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = einops.rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = einops.rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = einops.repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.mlp_head(x)

        x = einops.rearrange(x, 'a b -> b a').unsqueeze(0)

        return x


class FunPosTransformerModel(ViViT):

    def __init__(self):
        super().__init__(
                image_size = CONFIG[MODEL]['img_width'],
                patch_size = 16,
                num_frames = CONFIG[MODEL]['seq_len'],
                dim = 192,
                depth = 4,
                heads = 3,
                pool = 'cls',
                in_channels = 3,
                dim_head = 64,
                dropout = 0.1,
                emb_dropout = 0.1,
                scale_dim = 4)
