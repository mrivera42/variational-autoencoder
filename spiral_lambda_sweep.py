"""Spiral DDPM with a noise-scaler lambda: track a few sample points and show
where each STARTS (x_T) and ENDS (x_0) as lambda varies.

    x_{t-1} = mean + lambda * sigma_t * z      (lambda = 1 full noise, 0 none)

Same 5 starting points and the same per-step noise realization are used for every
lambda (only the multiplier differs), so differences are purely due to lambda.

    uv run python spiral_lambda_sweep.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import spiral_diffusion as sp          # reuse make_spiral, train, DEVICE

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
N = 5                                   # points to track
LAMBDAS = [float(v) for v in os.environ.get("LAMBDAS", "1.0,0.75,0.5,0.25,0.0").split(",")]
OUTNAME = os.environ.get("OUTNAME", "spiral_lambda_sweep.png")
Z_SEED = 7                              # fixes the per-step noise realization
X_SEED = 3                              # fixes the 5 starting points x_T
COLORS = plt.get_cmap("tab10").colors


@torch.no_grad()
def sample_paths(model, diffusion, x_init, lam):
    """Reverse process with noise scaled by lam. Returns trajectory [T+1, N, 2]."""
    torch.manual_seed(Z_SEED)           # same z sequence for every lambda
    x = x_init.clone()
    traj = [x.clone()]
    for step in reversed(range(diffusion.num_timesteps)):
        t = torch.full((N,), step, device=sp.DEVICE, dtype=torch.long)
        eps = model(x, t)
        betas_t = diffusion.betas[step]
        sqrt_1mab = diffusion.sqrt_one_minus_alpha_bars[step]
        sqrt_recip = diffusion.sqrt_recip_alphas[step]
        mean = sqrt_recip * (x - betas_t / sqrt_1mab * eps)
        if step > 0:
            var = diffusion.posterior_variance[step]
            z = torch.randn_like(x)
            x = mean + lam * torch.sqrt(var) * z
        else:
            x = mean
        traj.append(x.clone())
    return torch.stack(traj).cpu().numpy()


def main():
    data = sp.make_spiral(4000)
    print("training spiral DDPM (T=1000) ...")
    model, diffusion = sp.train(1000, data)

    torch.manual_seed(X_SEED)
    x_init = torch.randn(N, 2, device=sp.DEVICE)      # same 5 starts for all lambdas
    tg = data.numpy()

    fig, axes = plt.subplots(1, len(LAMBDAS), figsize=(3.2 * len(LAMBDAS), 3.4))
    for ax, lam in zip(axes, LAMBDAS):
        traj = sample_paths(model, diffusion, x_init, lam)     # [T+1, N, 2]
        ax.scatter(tg[:, 0], tg[:, 1], s=4, c="0.85", alpha=0.5, zorder=0)
        for i in range(N):
            end = traj[-1, i, :]                                           # generated point x_0
            ax.scatter(*end, marker="o", s=90, c=[COLORS[i]], edgecolors="k", lw=0.6, zorder=4)
        ax.set_title(f"$\\lambda$ = {lam:g}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)

    fig.suptitle("Swiss-roll DDPM: generated points as noise scale $\\lambda$ varies", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{OUTNAME}", dpi=150, bbox_inches="tight")
    print(f"saved figures/{OUTNAME}")


if __name__ == "__main__":
    main()
