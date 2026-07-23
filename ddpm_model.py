"""Denoiser networks for DDPM.

Two denoisers, both with the signature model(x, t) -> predicted noise:
  - UNet          : pixel-space, operates on [B, 1, 28, 28] images.
  - LatentDenoiser: latent-space MLP, operates on [B, latent_dim] VAE latents.

Both condition on the timestep t via a shared sinusoidal embedding.

NOTE (self-scaffold): the `forward` methods tagged `# SCAFFOLD` are the learning
exercises. To redo yourself, blank out those bodies (keep signatures + the
`__init__` layer definitions) and reimplement. See SCAFFOLD_NOTES.md.
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Map integer timesteps t (shape [B]) -> embeddings [B, dim].

    Same construction as Transformer positional encodings: interleave
    sin/cos of t scaled by a geometric range of frequencies.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # SCAFFOLD
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )                                             # [half]
        args = t[:, None].float() * freqs[None, :]    # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]


# ---------------------------------------------------------------------------
# Pixel-space denoiser: small UNet
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Conv -> add-time -> conv, with a residual skip.

    The time embedding is projected to `out_ch` and added to the feature map
    (broadcast over H, W) between the two convolutions.
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        # match channels on the residual path when in_ch != out_ch
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # SCAFFOLD
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]   # broadcast over H, W
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.res_conv(x)


class UNet(nn.Module):
    """Two-level UNet for 28x28 MNIST. Spatial path: 28 -> 14 -> 7 -> 14 -> 28.

    Skip connections concatenate the matching-resolution encoder feature onto
    the decoder input before each up-block.
    """

    def __init__(self, in_ch=1, base_ch=64, time_dim=128):
        super().__init__()

        # timestep MLP: sinusoidal embedding -> 2-layer MLP
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # encoder
        self.down1 = ResBlock(base_ch, base_ch, time_dim)
        self.downsample1 = nn.Conv2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)      # 28 -> 14
        self.down2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)  # 14 -> 7

        # bottleneck
        self.mid = ResBlock(base_ch * 2, base_ch * 2, time_dim)

        # decoder (in_ch of each up-block includes the concatenated skip channels)
        self.upsample2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.up2 = ResBlock(base_ch * 2 + base_ch * 2, base_ch, time_dim)
        self.upsample1 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)          # 14 -> 28
        self.up1 = ResBlock(base_ch + base_ch, base_ch, time_dim)

        self.final = nn.Conv2d(base_ch, in_ch, kernel_size=1)

    def forward(self, x, t):
        # SCAFFOLD
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)
        s1 = self.down1(x, t_emb)          # [B, base, 28, 28]
        x = self.downsample1(s1)           # [B, base, 14, 14]
        s2 = self.down2(x, t_emb)          # [B, 2base, 14, 14]
        x = self.downsample2(s2)           # [B, 2base, 7, 7]

        x = self.mid(x, t_emb)

        x = self.upsample2(x)                              # [B, 2base, 14, 14]
        x = self.up2(torch.cat([x, s2], dim=1), t_emb)
        x = self.upsample1(x)                              # [B, base, 28, 28]
        x = self.up1(torch.cat([x, s1], dim=1), t_emb)
        return self.final(x)


# ---------------------------------------------------------------------------
# Latent-space denoiser: MLP over VAE latents
# ---------------------------------------------------------------------------
class LatentDenoiser(nn.Module):
    """MLP denoiser for latent DDPM. Predicts noise in the VAE latent space.

    Mirrors the MLP style of your VAE. The timestep is embedded and injected by
    adding a projection of t_emb to the hidden activation.
    """

    def __init__(self, latent_dim, hidden_dim=256, time_dim=128, n_layers=3):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.SiLU()

    def forward(self, z, t):
        # SCAFFOLD
        t_emb = self.time_mlp(t)                                # [B, time_dim]
        h = self.act(self.in_proj(z)) + self.time_proj(t_emb)  # [B, hidden_dim]
        for block in self.blocks:
            h = h + self.act(block(h))                          # residual MLP
        return self.out_proj(h)


if __name__ == "__main__":
    # quick shape sanity check
    B = 4
    t = torch.randint(0, 1000, (B,))

    unet = UNet(in_ch=1, base_ch=64, time_dim=128)
    x = torch.randn(B, 1, 28, 28)
    print("UNet out:", unet(x, t).shape)          # expect [4, 1, 28, 28]

    mlp = LatentDenoiser(latent_dim=20)
    z = torch.randn(B, 20)
    print("LatentDenoiser out:", mlp(z, t).shape)  # expect [4, 20]
