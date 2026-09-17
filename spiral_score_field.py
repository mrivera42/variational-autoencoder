"""Score / denoising vector field of the spiral DDPM.

On an evenly-spaced grid over the plane, draw the model's predicted push toward
the data: the score  s = -eps_theta(x,t) / sqrt(1 - alpha_bar_t)  (direction the
reverse process moves a point), arrow length + color = magnitude. Shown at a few
noise levels t so you see the field sharpen onto the spiral as t -> 0.

Runs on CPU (small model) so it won't contend with a GPU job.
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


def train(diffusion, data, iters=3500, batch=512, lr=1e-3):
    torch.manual_seed(0)
    model = LatentDenoiser(latent_dim=2, hidden_dim=128, time_dim=64, n_layers=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = data.shape[0]
    for it in range(iters):
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
    torch.set_grad_enabled(False)                                    # eval only from here

    G = 22
    g = np.linspace(-3, 3, G)
    X, Y = np.meshgrid(g, g)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32, device=DEVICE)
    tg = data.numpy()
    t_list = [999, 600, 300, 50]

    fig, axes = plt.subplots(1, len(t_list), figsize=(3.4 * len(t_list), 3.6))
    for ax, tv in zip(axes, t_list):
        eps = model(pts, torch.full((pts.shape[0],), tv, device=DEVICE, dtype=torch.long))
        score = -eps / diffusion.sqrt_one_minus_alpha_bars[tv]        # points toward data
        U = score[:, 0].numpy().reshape(G, G)
        V = score[:, 1].numpy().reshape(G, G)
        M = np.sqrt(U ** 2 + V ** 2)
        ax.scatter(tg[:, 0], tg[:, 1], s=3, c="0.8", alpha=0.5, zorder=0)
        ax.quiver(X, Y, U, V, M, cmap="viridis", zorder=1, width=0.006)
        ax.set_title(f"t = {tv}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)

    fig.suptitle("Spiral DDPM: predicted denoising vector field (score) at decreasing noise", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG}/spiral_score_field.png", dpi=150, bbox_inches="tight")
    print("saved figures/spiral_score_field.png")


if __name__ == "__main__":
    main()
