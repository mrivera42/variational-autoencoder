"""Quantify the collapse as the noise scale lambda decreases.

For each lambda we generate M samples and compute two metrics:
  - Sample diversity   = mean pairwise distance among generated points
                         (-> 0 when they collapse to a single location)
  - Energy distance    = sqrt(2 E|X-Y| - E|X-X'| - E|Y-Y'|) between generated (X)
                         and real data (Y); ~0 when they match, large on collapse.
Both are plotted vs lambda so you can see WHERE the collapse happens.

    uv run python spiral_collapse_metric.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import spiral_diffusion as sp

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
M = 1500
LAMBDAS = np.linspace(0.0, 1.0, 31)


@torch.no_grad()
def generate(model, diffusion, m, lam, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, 2, device=sp.DEVICE)
    for step in reversed(range(diffusion.num_timesteps)):
        t = torch.full((m,), step, device=sp.DEVICE, dtype=torch.long)
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
    return x


def mpd(a, b):                                   # mean pairwise distance
    return torch.cdist(a, b).mean().item()


def main():
    data = sp.make_spiral(4000)
    print("training spiral DDPM (T=1000) ...")
    model, diffusion = sp.train(1000, data)

    real = data[torch.randperm(data.shape[0])[:M]].to(sp.DEVICE)
    d_BB = mpd(real, real)

    diversity, energy = [], []
    for lam in LAMBDAS:
        g = generate(model, diffusion, M, float(lam))
        d_AA = mpd(g, g)
        d_AB = mpd(g, real)
        diversity.append(d_AA)
        energy.append(np.sqrt(max(0.0, 2 * d_AB - d_AA - d_BB)))
        print(f"  lambda={lam:.3f}  diversity={d_AA:.3f}  energy={energy[-1]:.3f}")

    diversity, energy = np.array(diversity), np.array(energy)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, y, ylab, color in [
        (axes[0], diversity, "mean pairwise distance", "tab:blue"),
        (axes[1], energy, "energy distance to data", "tab:red"),
    ]:
        ax.plot(LAMBDAS, y, marker="o", ms=4, color=color)
        ax.axvspan(0.0, 0.15, color="0.9", zorder=0)          # highlight the collapse region
        ax.set_xlabel("noise scale $\\lambda$")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()                                     # lambda decreasing left -> right
    axes[0].set_title("Sample diversity")
    axes[1].set_title("Distribution mismatch")
    fig.suptitle("Swiss-roll DDPM: collapse metrics vs $\\lambda$", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{FIG}/spiral_collapse_metric.png", dpi=150, bbox_inches="tight")
    print("saved figures/spiral_collapse_metric.png")


if __name__ == "__main__":
    main()
