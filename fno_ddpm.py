"""Neural-operator DDPM denoiser: an FNO-style network built from SpectralConv2d.

Same signature as the UNet denoiser -- model(x_t, t) -> predicted noise -- so it
plugs straight into GaussianDiffusion. But every layer is resolution-invariant
(spectral convs + pointwise 1x1 convs + GroupNorm + pointwise time injection),
so the SAME trained model can be sampled at any resolution.

This is the Camp-1 "neural-operator diffusion" idea (cf. DDO, infinity-Diff):
replace the fixed-grid UNet denoiser with a discretization-invariant operator.
"""

import torch
import torch.nn as nn

from ddpm_model import SinusoidalTimeEmbedding
from spectral_conv import SpectralConv2d


class FNOBlock(nn.Module):
    """FNO block: spectral-conv + pointwise skip, + time embedding, residual."""

    def __init__(self, width, im_shape, ksize, time_dim):
        super().__init__()
        self.spectral = SpectralConv2d.from_scratch(width, width, im_shape, ksize=ksize)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)     # 1x1 -> res-invariant
        self.time_proj = nn.Linear(time_dim, width)
        self.norm = nn.GroupNorm(8, width)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.spectral(x) + self.pointwise(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]            # broadcast over H, W
        h = self.act(self.norm(h))
        return x + h                                               # residual


class FNODenoiser(nn.Module):
    """Resolution-invariant DDPM denoiser. model(x, t) -> predicted noise."""

    def __init__(self, in_ch=1, width=48, n_blocks=4, im_shape=(28, 28), ksize=15, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.lift = nn.Conv2d(in_ch, width, kernel_size=1)         # pointwise lift
        self.blocks = nn.ModuleList(
            [FNOBlock(width, im_shape, ksize, time_dim) for _ in range(n_blocks)]
        )
        self.proj = nn.Conv2d(width, in_ch, kernel_size=1)         # pointwise project

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.lift(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.proj(h)


if __name__ == "__main__":
    m = FNODenoiser()
    for res in (28, 40, 56):
        x = torch.randn(2, 1, res, res)
        t = torch.randint(0, 1000, (2,))
        print(f"{res}x{res} -> {tuple(m(x, t).shape)}")   # same-resolution output at any res
    print("params:", sum(p.numel() for p in m.parameters()))
