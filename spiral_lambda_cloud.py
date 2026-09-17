"""Spiral DDPM: full generated point cloud (n=4000) as the noise scale lambda
varies. Same setup as spiral_lambda_sweep.py but scatters the whole generated
distribution per lambda instead of a few tracked points.

    uv run python spiral_lambda_cloud.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import spiral_diffusion as sp

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
N = 4000
LAMBDAS = [1.0, 0.75, 0.5, 0.25, 0.0]
SEED = 7


@torch.no_grad()
def generate(model, diffusion, lam):
    torch.manual_seed(SEED)                          # same noise realization per lambda
    x = torch.randn(N, 2, device=sp.DEVICE)
    for step in reversed(range(diffusion.num_timesteps)):
        t = torch.full((N,), step, device=sp.DEVICE, dtype=torch.long)
        eps = model(x, t)
        betas_t = diffusion.betas[step]
        sqrt_1mab = diffusion.sqrt_one_minus_alpha_bars[step]
        sqrt_recip = diffusion.sqrt_recip_alphas[step]
        mean = sqrt_recip * (x - betas_t / sqrt_1mab * eps)
        if step > 0:
            var = diffusion.posterior_variance[step]
            x = mean + lam * torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mean
    return x.cpu().numpy()


def main():
    data = sp.make_spiral(4000)
    print("training spiral DDPM (T=1000) ...")
    model, diffusion = sp.train(1000, data)
    tg = data.numpy()

    fig, axes = plt.subplots(1, len(LAMBDAS), figsize=(3.2 * len(LAMBDAS), 3.4))
    for ax, lam in zip(axes, LAMBDAS):
        pts = generate(model, diffusion, lam)
        ax.scatter(tg[:, 0], tg[:, 1], s=4, c="0.85", alpha=0.4, zorder=0)      # data
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c="crimson", alpha=0.25, zorder=1)  # generated
        ax.set_title(f"$\\lambda$ = {lam:g}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)

    fig.suptitle("Swiss-roll DDPM: 4000 generated points as noise scale $\\lambda$ varies", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG}/spiral_lambda_cloud.png", dpi=150, bbox_inches="tight")
    print("saved figures/spiral_lambda_cloud.png")


if __name__ == "__main__":
    main()
