"""Graph-neural-operator DDPM denoiser (from user's gno.py, bug fixed).

Treats the image as a point cloud on a normalized [0,1]^2 grid; a GNO kernel
integrates over neighbors within a fixed normalized radius. Normalized coords +
fixed radius => resolution-invariant (train at one grid, sample at another).

Fix vs. the original GNOLayer: `out = out.scatter_add(...)` is now ASSIGNED (the
original discarded it, so the nonlocal/kernel path was dead), and the erroneous
`.squeeze(2)` is removed. Edges are cached per resolution to keep sampling fast.
"""

import torch
import torch.nn as nn

from ddpm_model import SinusoidalTimeEmbedding


def grid_coords(H, W, device):
    xs = torch.linspace(0, 1, H, device=device)
    ys = torch.linspace(0, 1, W, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.flatten(), gy.flatten()], dim=1)      # [n, 2]


class GNOLayer(nn.Module):
    def __init__(self, radius, in_ch, out_ch, offset_dim=2):
        super().__init__()
        self.radius, self.in_ch, self.out_ch = radius, in_ch, out_ch
        self.kernel = nn.Sequential(
            nn.Linear(offset_dim, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, in_ch * out_ch),
        )
        self.w = nn.Linear(in_ch, out_ch)

    def forward(self, features, edges):
        i_idx, j_idx, offsets, counts = edges
        k = self.kernel(offsets).reshape(offsets.shape[0], self.out_ch, self.in_ch)
        f = features[:, j_idx, :]                                  # [b, E, in]
        p = torch.einsum("eoc,bec->beo", k, f)                    # [b, E, out]
        out = torch.zeros(features.shape[0], features.shape[1], self.out_ch, device=features.device)
        idx = i_idx.view(1, -1, 1).expand_as(p)
        out = out.scatter_add(1, idx, p)                          # FIX: assigned
        out = out / counts.clamp(min=1).view(1, -1, 1)
        return self.w(features) + out                             # FIX: no squeeze


class GNOBlock(nn.Module):
    def __init__(self, radius, width, time_dim):
        super().__init__()
        self.gno = GNOLayer(radius, width, width)
        self.time_proj = nn.Linear(time_dim, width)
        self.norm = nn.LayerNorm(width)
        self.act = nn.GELU()

    def forward(self, features, t_emb, edges):
        h = self.gno(features, edges) + self.time_proj(t_emb).unsqueeze(1)
        return features + self.act(self.norm(h))


class GNODenoiser(nn.Module):
    def __init__(self, radius=0.1, width=24, n_blocks=3, time_dim=128):
        super().__init__()
        self.radius = radius
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.lift = nn.Linear(1, width)
        self.blocks = nn.ModuleList([GNOBlock(radius, width, time_dim) for _ in range(n_blocks)])
        self.proj = nn.Linear(width, 1)
        self._cache = {}

    def _edges(self, H, W, device):
        if (H, W) not in self._cache:
            coords = grid_coords(H, W, device)
            d = torch.cdist(coords, coords)
            i_idx, j_idx = torch.where(d < self.radius)
            offsets = coords[i_idx] - coords[j_idx]
            counts = torch.zeros(coords.shape[0], device=device).scatter_add_(
                0, i_idx, torch.ones_like(i_idx, dtype=torch.float))
            self._cache[(H, W)] = (i_idx, j_idx, offsets, counts)
        return self._cache[(H, W)]

    def forward(self, x, t):
        B, C, H, W = x.shape
        edges = self._edges(H, W, x.device)
        feats = x.permute(0, 2, 3, 1).reshape(B, H * W, C)        # [B, n, 1]
        t_emb = self.time_mlp(t)
        h = self.lift(feats)
        for block in self.blocks:
            h = block(h, t_emb, edges)
        out = self.proj(h)                                        # [B, n, 1]
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


if __name__ == "__main__":
    m = GNODenoiser(radius=0.1, width=24, n_blocks=3)
    for res in (28, 56):
        x = torch.randn(2, 1, res, res); t = torch.randint(0, 1000, (2,))
        print(f"{res}: {tuple(m(x, t).shape)}")
    print("params:", sum(p.numel() for p in m.parameters()))
