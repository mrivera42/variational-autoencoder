"""Watch a Swiss-roll DDPM LEARN the reverse process over the course of training.

Top row : samples drawn by running the full reverse process (x_T ~ N(0,I) -> x_0)
          using the model as it looks at increasing training iterations. Early on
          the reverse process can't undo the noise, so samples are a blob; as
          training proceeds the samples collapse onto the swiss-roll manifold.
          The same random x_T seed is used at every snapshot so panels are
          directly comparable.
Bottom  : the training loss (noise-prediction MSE) vs iteration, with the
          snapshot iterations marked.

Reuses the exact same machinery as the rest of the repo:
  - GaussianDiffusion (diffusion.py)  for the forward/reverse math
  - LatentDenoiser    (ddpm_model.py) with latent_dim=2 as the denoiser

    uv run python spiral_training_viz.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion import GaussianDiffusion
from ddpm_model import LatentDenoiser

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
DEVICE = "cpu"

T = 1000
N_ITERS = 4000
SNAPSHOT_ITERS = [0, 30, 80, 200, 500, 1200, 2500, 4000]   # 0 = untrained model
N_SAMPLES = 2000
SAMPLE_SEED = 7                              # fixed x_T across snapshots


def make_spiral(n, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    pts = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
    pts += 0.30 * rng.standard_normal(pts.shape)
    pts = (pts - pts.mean(0)) / pts.std(0)          # normalize -> matches N(0,I) prior
    return torch.tensor(pts, dtype=torch.float32)


@torch.no_grad()
def sample(model, diffusion, n, seed=SAMPLE_SEED):
    """Full reverse process from a fixed noise seed -> [n, 2] generated points."""
    torch.manual_seed(seed)
    x = torch.randn(n, 2, device=DEVICE)
    for step in reversed(range(diffusion.num_timesteps)):
        t = torch.full((n,), step, device=DEVICE, dtype=torch.long)
        x = diffusion.p_sample(model, x, t)
    return x.cpu().numpy()


def train_with_snapshots(data):
    torch.manual_seed(0)
    diffusion = GaussianDiffusion(num_timesteps=T, beta_start=1e-4, beta_end=0.02, device=DEVICE)
    model = LatentDenoiser(latent_dim=2, hidden_dim=128, time_dim=64, n_layers=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    N = data.shape[0]
    losses = []
    snapshots = {}   # iter -> generated points [n, 2]

    want = set(SNAPSHOT_ITERS)
    if 0 in want:    # snapshot the untrained model before any gradient step
        snapshots[0] = sample(model, diffusion, N_SAMPLES)
        print("  snapshot @ iter 0 (untrained)")

    for it in range(N_ITERS):
        idx = torch.randint(0, N, (512,))
        x0 = data[idx].to(DEVICE)
        t = torch.randint(0, T, (512,), device=DEVICE)
        loss = diffusion.p_losses(model, x0, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        done = it + 1
        if done in want:
            snapshots[done] = sample(model, diffusion, N_SAMPLES)
            print(f"  snapshot @ iter {done}  (loss {loss.item():.4f})")

    return diffusion, losses, snapshots


def smooth(x, k=50):
    """Simple running mean to make the noisy per-iter loss readable."""
    if len(x) < k:
        return np.asarray(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def make_figure(data, losses, snapshots):
    order = SNAPSHOT_ITERS
    K = len(order)
    tg = data.numpy()

    fig = plt.figure(figsize=(2.5 * K, 5.6))
    gs = fig.add_gridspec(2, K, height_ratios=[1.15, 0.85], hspace=0.35, wspace=0.1)

    # --- top row: reverse-process samples at each snapshot ---
    for j, it in enumerate(order):
        ax = fig.add_subplot(gs[0, j])
        ax.scatter(tg[:, 0], tg[:, 1], s=3, c="0.8", alpha=0.5, zorder=0)          # data
        pts = snapshots[it]
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c="crimson", alpha=0.35, zorder=1)   # generated
        label = "untrained" if it == 0 else f"iter {it}"
        ax.set_title(label, fontsize=11)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        if j == 0:
            ax.set_ylabel("reverse-process\nsamples", fontsize=10)

    # --- bottom: training loss spanning full width ---
    axl = fig.add_subplot(gs[1, :])
    axl.plot(range(1, len(losses) + 1), losses, color="steelblue", alpha=0.25, lw=0.8,
             label="loss (per iter)")
    sm = smooth(losses, k=50)
    axl.plot(range(1, len(sm) + 1), sm, color="navy", lw=1.8, label="loss (smoothed)")
    axl.set_xscale("log")                          # early snapshots crowd near 0 on a linear axis
    for it in order:
        if it == 0:
            continue
        axl.axvline(it, color="crimson", ls="--", lw=1.0, alpha=0.7)
        axl.annotate(f"{it}", xy=(it, axl.get_ylim()[1]), xytext=(0, -2),
                     textcoords="offset points", ha="center", va="top",
                     fontsize=8, color="crimson")
    axl.set_xlabel("training iteration (log scale)")
    axl.set_ylabel("noise-prediction MSE")
    axl.set_title("Training loss (snapshots marked)", fontsize=11)
    axl.grid(True, alpha=0.3)
    axl.legend(loc="upper right", fontsize=9)

    fig.suptitle("Swiss-roll DDPM learning the reverse process over training",
                 y=0.98, fontsize=13)
    fig.savefig(f"{FIG}/spiral_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/spiral_training.png")


def main():
    data = make_spiral(4000)
    diffusion, losses, snapshots = train_with_snapshots(data)
    make_figure(data, losses, snapshots)
    print("ALL DONE")


if __name__ == "__main__":
    main()
