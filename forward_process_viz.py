"""Visualize the FORWARD diffusion process: data distribution -> N(0, I).

No trained model needed — the forward process is the closed form
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
so we just apply GaussianDiffusion.q_sample at increasing timesteps t.

2D swiss roll point cloud dissolving into a Gaussian blob as t increases.

    uv run python forward_process_viz.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion import GaussianDiffusion

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
DEVICE = "cpu"


def make_spiral(n, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    pts = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
    pts += 0.30 * rng.standard_normal(pts.shape)
    pts = (pts - pts.mean(0)) / pts.std(0)
    return torch.tensor(pts, dtype=torch.float32)


def main():
    diffusion = GaussianDiffusion(num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device=DEVICE)
    t_list = [0, 50, 150, 400, 700, 999]

    swiss = make_spiral(4000)

    fig, axes = plt.subplots(1, len(t_list), figsize=(2.4 * len(t_list), 2.9))

    for j, t in enumerate(t_list):
        sa = diffusion.sqrt_alpha_bars[t].item()      # signal fraction

        # --- 2D swiss roll ---
        tt = torch.full((swiss.shape[0],), t, dtype=torch.long)
        xt = diffusion.q_sample(swiss, tt).numpy()
        ax = axes[j]
        ax.scatter(xt[:, 0], xt[:, 1], s=2, alpha=0.3, c="crimson")
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(f"t = {t}\n" + r"$\sqrt{\bar\alpha}$" + f" = {sa:.2f}", fontsize=10)

    axes[0].set_ylabel("2D swiss roll", fontsize=11)
    fig.suptitle(r"Forward diffusion: data distribution $\rightarrow$ $\mathcal{N}(0, I)$ as $t$ increases",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{FIG}/forward_process.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/forward_process.png")


if __name__ == "__main__":
    main()
