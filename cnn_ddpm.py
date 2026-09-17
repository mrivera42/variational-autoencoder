"""Pooling-free CNN DDPM denoiser + conversion to spectral (FNO) at inference.

This is the "convert" method: train an ordinary CNN denoiser (circular-padded
convs so the convolution-theorem conversion is exact), then at inference replace
each spatial conv with a SpectralConv2d built from it (user's `kernel_to_spectral`).
Because spectral convs are resolution-invariant, the converted model can sample at
resolutions the CNN never trained on.

Architecture mirrors FNODenoiser (pointwise lift/proj, GroupNorm, time embedding,
residual) so the only change on conversion is Conv2d -> SpectralConv2d.
"""

import copy

import torch
import torch.nn as nn

from ddpm_model import SinusoidalTimeEmbedding
from spectral_conv import SpectralConv2d


class ConvBlock(nn.Module):
    def __init__(self, width, ksize, time_dim):
        super().__init__()
        # circular padding -> the FFT-based conversion is exact at train resolution
        self.spatial = nn.Conv2d(width, width, ksize, padding="same", padding_mode="circular")
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.time_proj = nn.Linear(time_dim, width)
        self.norm = nn.GroupNorm(8, width)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.spatial(x) + self.pointwise(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm(h))
        return x + h


class CNNDenoiser(nn.Module):
    """Pooling-free CNN denoiser. model(x, t) -> predicted noise."""

    def __init__(self, in_ch=1, width=48, n_blocks=4, ksize=7, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.lift = nn.Conv2d(in_ch, width, kernel_size=1)
        self.blocks = nn.ModuleList([ConvBlock(width, ksize, time_dim) for _ in range(n_blocks)])
        self.proj = nn.Conv2d(width, in_ch, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.lift(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.proj(h)


def convert_to_spectral(cnn, im_shape=(28, 28)):
    """Return a copy of the trained CNN with each spatial Conv2d replaced by its
    spectral (Fourier) equivalent via the convolution theorem — resolution-invariant."""
    model = copy.deepcopy(cnn)
    for block in model.blocks:
        block.spatial = SpectralConv2d(block.spatial, im_shape)
    return model


if __name__ == "__main__":
    m = CNNDenoiser()
    x = torch.randn(2, 1, 28, 28); t = torch.randint(0, 1000, (2,))
    print("cnn out:", tuple(m(x, t).shape))
    s = convert_to_spectral(m)
    for res in (28, 40, 56):
        xi = torch.randn(2, 1, res, res)
        print(f"converted @ {res}:", tuple(s(xi, t).shape))
    # sanity: converted ~= cnn at native resolution (circular conv)
    with torch.no_grad():
        diff = (m(x, t) - s(x, t)).abs().max().item()
    print("max|cnn - converted| @28:", diff)
