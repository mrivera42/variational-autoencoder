"""Score field v2: uniform-length arrows (direction crisp everywhere) colored by
magnitude, with the noise level labeled. Reveals the field pointing ONTO the
spiral arms at low noise (small t). Saves a separate figure.

Runs on CPU. Does not overwrite spiral_score_field.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion import GaussianDiffusion
from ddpm_model import LatentDenoiser

FIG = "figures"; os.makedirs(FIG, exist_ok=True)
DEVICE = "cpu"


def make_spiral(n, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    pts = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
    pts += 0.30 * rng.standard_normal(pts.shape)
    pts = (pts - pts.mean(0)) / pts.std(0)
    return torch.tensor(pts, dtype=torch.float32)


def train(diffusion, data, iters=5000, batch=512, lr=1e-3):
    torch.manual_seed(0)
    model = LatentDenoiser(latent_dim=2, hidden_dim=128, time_dim=64, n_layers=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = data.shape[0]
    for _ in range(iters):
        x0 = data[torch.randint(0, N, (batch,))].to(DEVICE)
        t = torch.randint(0, diffusion.num_timesteps, (batch,), device=DEVICE)
        loss = diffusion.p_losses(model, x0, t)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    diffusion = GaussianDiffusion(1000, beta_start=1e-4, beta_end=0.02, device=DEVICE)
    data = make_spiral(4000)
    print("training spiral model (cpu) ...")
    model = train(diffusion, data)
    torch.set_grad_enabled(False)

    G = 28
    g = np.linspace(-3, 3, G)
    X, Y = np.meshgrid(g, g)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32, device=DEVICE)
    tg = data.numpy()
    t_list = [800, 400, 150, 30]

    fig, axes = plt.subplots(1, len(t_list), figsize=(3.5 * len(t_list), 3.8))
    for ax, tv in zip(axes, t_list):
        eps = model(pts, torch.full((pts.shape[0],), tv, device=DEVICE, dtype=torch.long))
        score = (-eps / diffusion.sqrt_one_minus_alpha_bars[tv]).numpy()
        M = np.hypot(score[:, 0], score[:, 1])
        Un = (score[:, 0] / (M + 1e-8)).reshape(G, G)          # unit direction
        Vn = (score[:, 1] / (M + 1e-8)).reshape(G, G)
        sa = diffusion.sqrt_alpha_bars[tv].item()              # signal fraction
        ax.scatter(tg[:, 0], tg[:, 1], s=3, c="0.8", alpha=0.5, zorder=0)
        ax.quiver(X, Y, Un, Vn, M.reshape(G, G), cmap="viridis",
                  pivot="mid", scale=32, width=0.005, zorder=1)
        ax.set_title(f"t = {tv}    " + r"$\sqrt{\bar\alpha}$" + f" = {sa:.2f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)

    fig.suptitle(r"Spiral DDPM denoising field: arrow = direction, color = strength   "
                 r"($\sqrt{\bar\alpha}$ = signal left; low = high noise)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/spiral_score_field_direction.png", dpi=150, bbox_inches="tight")
    print("saved figures/spiral_score_field_direction.png")


if __name__ == "__main__":
    main()
