import torch
import einops

from torch import nn, einsum
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from modules.transformer import FSATransformerEncoder

from utils.config import CONFIG

MODEL='model2'


class ViViTBackbone(nn.Module):

    def __init__(self, t, h, w, patch_t, patch_h, patch_w, dim, depth, heads, mlp_dim, dim_head=3,
                 channels=3, mode='tubelet', emb_dropout=0., dropout=0.):
        super().__init__()

        assert t % patch_t == 0 and h % patch_h == 0 and w % patch_w == 0, \
            "Video dimensions should be divisible by tubelet size"

        self.T = t
        self.H = h
        self.W = w
        self.channels = channels
        self.t = patch_t
        self.h = patch_h
        self.w = patch_w
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nt = self.T // self.t
        self.nh = self.H // self.h
        self.nw = self.W // self.w

        tubelet_dim = self.t * self.h * self.w * channels

        self.to_tubelet_embedding = nn.Sequential(
            Rearrange('b (t pt) c (h ph) (w pw) -> b t (h w) (pt ph pw c)', pt=self.t, ph=self.h, pw=self.w),
            nn.Linear(tubelet_dim, dim)
        )

        # repeat same spatial position encoding temporally
        self.pos_embedding = nn.parameter.Parameter(torch.randn(1, 1, self.nh * self.nw, dim)).repeat(1, self.nt, 1, 1).to(self.device)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = FSATransformerEncoder(dim, depth, heads, dim_head, mlp_dim, self.nt, self.nh, self.nw, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//self.t),
            nn.Linear(dim//self.t, self.T)
        )

    def forward(self, x):
        """ x is a video frame sequence: (b, T, C, H, W) """

        tokens = self.to_tubelet_embedding(x)

        tokens += self.pos_embedding
        tokens = self.dropout(tokens)

        x = self.transformer(tokens)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = einops.rearrange(x, 'a b -> b a').unsqueeze(0)
        return x


class FunPosTransformerModel(ViViTBackbone):

    def __init__(self):
        super().__init__(
                t = CONFIG[MODEL]['seq_len'],
                h = CONFIG[MODEL]['img_height'],
                w = CONFIG[MODEL]['img_width'],
                patch_t = 8,
                patch_h = 16,
                patch_w = 16,
                dim = 768,
                depth = 12,
                heads = 12,
                mlp_dim = 4096,
        )
