"""DDPM on a 2D Swiss-roll (spiral), with visualizations of the reverse-process
trajectories — the whole point being that in 2D you can literally *see* each
point travel from noise back to the data manifold, step by step.

Reuses the exact same machinery as the MNIST work:
  - GaussianDiffusion (diffusion.py) for the forward/reverse math + add_noise flag
  - LatentDenoiser   (ddpm_model.py) as the denoiser, here with latent_dim=2
    i.e. an MLP that takes a 2D point x and a timestep t and predicts the noise.

Figures written to figures/:
  spiral_samples.png            data vs learned samples (sanity)
  spiral_traj_noise.png         reverse trajectories: with vs without noise term
  spiral_traj_T.png             reverse trajectories as T (number of steps) varies

    uv run python spiral_diffusion.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import torch

from diffusion import GaussianDiffusion
from ddpm_model import LatentDenoiser

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
DEVICE = "cpu"          # data is 2D; CPU is fastest and keeps things deterministic


# --------------------------------------------------------------------------
# data: 2D swiss roll (the (x, z) projection of the classic 3D swiss roll)
# --------------------------------------------------------------------------
def make_spiral(n, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))       # angle along the roll
    x = t * np.cos(t)
    y = t * np.sin(t)
    pts = np.stack([x, y], axis=1)
    pts += 0.30 * rng.standard_normal(pts.shape)    # a little thickness
    pts = (pts - pts.mean(0)) / pts.std(0)          # normalize -> matches N(0,I) prior
    return torch.tensor(pts, dtype=torch.float32)


# --------------------------------------------------------------------------
# training (a few thousand steps of the standard DDPM loss)
# --------------------------------------------------------------------------
def train(T, data, n_iters=4000, batch=512, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    diffusion = GaussianDiffusion(num_timesteps=T, beta_start=1e-4, beta_end=0.02, device=DEVICE)
    model = LatentDenoiser(latent_dim=2, hidden_dim=128, time_dim=64, n_layers=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    N = data.shape[0]
    for it in range(n_iters):
        idx = torch.randint(0, N, (batch,))
        x0 = data[idx].to(DEVICE)
        t = torch.randint(0, T, (batch,), device=DEVICE)
        loss = diffusion.p_losses(model, x0, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 1000 == 0:
            print(f"  [T={T}] iter {it+1}/{n_iters}  loss {loss.item():.4f}")
    return model, diffusion


# --------------------------------------------------------------------------
# reverse process, recording the full trajectory of every point
# --------------------------------------------------------------------------
@torch.no_grad()
def sample_trajectory(model, diffusion, n, add_noise, seed=0):
    torch.manual_seed(seed)                          # same x_T across settings
    x = torch.randn(n, 2, device=DEVICE)
    traj = [x.clone()]
    for step in reversed(range(diffusion.num_timesteps)):
        t = torch.full((n,), step, device=DEVICE, dtype=torch.long)
        x = diffusion.p_sample(model, x, t, add_noise=add_noise)
        traj.append(x.clone())
    return torch.stack(traj).cpu().numpy()           # [steps+1, n, 2]


# --------------------------------------------------------------------------
# plotting helpers
# --------------------------------------------------------------------------
def style_axes(ax, lim=3.0, mark_origin=True):
    """Label axes (x1, x2), draw light axis lines through 0, and mark the origin.

    The swiss roll is normalized to zero mean (see make_spiral), so the origin
    IS the data mean E[x_0] — hence worth marking explicitly.
    """
    ax.axhline(0, color="0.55", lw=0.6, zorder=0.5)
    ax.axvline(0, color="0.55", lw=0.6, zorder=0.5)
    if mark_origin:
        ax.scatter([0], [0], marker="*", s=120, c="gold", edgecolors="k",
                   lw=0.7, zorder=6, label="origin (data mean)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=9); ax.set_ylabel(r"$x_2$", fontsize=9)
    ax.tick_params(labelsize=8)


def draw_trajectories(ax, traj, target, title, n_show=8):
    """target scatter + generated endpoints + a few time-colored trajectories."""
    tg = target.numpy()
    ax.scatter(tg[:, 0], tg[:, 1], s=4, c="0.82", alpha=0.5, label="data", zorder=0)

    final = traj[-1]                                  # [n, 2] generated points
    ax.scatter(final[:, 0], final[:, 1], s=5, c="lightcoral", alpha=0.22,
               label="generated", zorder=1)

    S = traj.shape[0]
    for j in range(min(n_show, traj.shape[1])):
        path = traj[:, j, :]                          # [S, 2]
        segs = np.stack([path[:-1], path[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", array=np.arange(S - 1),
                            linewidth=1.1, alpha=0.8, zorder=2)
        ax.add_collection(lc)
        ax.scatter(*path[0], s=26, c="black", marker="x", lw=1.6, zorder=4)   # start (x_T)
        ax.scatter(*path[-1], s=26, c="red", marker="o",
                   edgecolors="k", lw=0.5, zorder=4)                          # end (x_0)

    ax.set_title(title, fontsize=11)
    style_axes(ax)


def fig_samples(model, diffusion, data):
    traj = sample_trajectory(model, diffusion, n=2000, add_noise=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    tg = data.numpy()
    ax.scatter(tg[:, 0], tg[:, 1], s=5, c="0.8", label="data (swiss roll)")
    ax.scatter(traj[-1][:, 0], traj[-1][:, 1], s=5, c="crimson", alpha=0.5, label="DDPM samples")
    ax.set_title("2D Swiss-roll DDPM: data vs. generated samples")
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/spiral_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved figures/spiral_samples.png")


def fig_noise_on_off(model, diffusion, data, n_show=8):
    """Several reverse trajectories per panel, over a faint cloud of all endpoints.

    Same `x_T` starts are used in both panels (shared seed), so you can compare
    how identical noise seeds evolve WITH vs WITHOUT the stochastic term. In the
    deterministic panel, watch whether distinct starts collapse onto the same
    endpoint(s) or spread across the manifold.
    """
    on = sample_trajectory(model, diffusion, n=2000, add_noise=True, seed=1)
    off = sample_trajectory(model, diffusion, n=2000, add_noise=False, seed=1)
    tg = data.numpy()
    colors = plt.cm.tab10(np.arange(n_show) % 10)             # one color per trajectory
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, traj, title in [(axes[0], on, "WITH noise term  (standard DDPM)"),
                            (axes[1], off, "WITHOUT noise term  (deterministic)")]:
        ax.scatter(tg[:, 0], tg[:, 1], s=6, c="0.82", alpha=0.55, zorder=0)
        ax.scatter(traj[-1][:, 0], traj[-1][:, 1], s=6, c="lightcoral", alpha=0.16,
                   zorder=1)                                   # faint cloud of all x_0
        for k in range(n_show):
            path = traj[:, k, :]                              # k-th sample's path
            segs = np.stack([path[:-1], path[1:]], axis=1)
            lc = LineCollection(segs, colors=colors[k], linewidth=1.3, alpha=0.85, zorder=2)
            ax.add_collection(lc)
            ax.scatter(*path[0], s=55, color=colors[k], marker="x", lw=2.0, zorder=4)   # start x_T
            ax.scatter(*path[-1], s=55, color=colors[k], marker="o", edgecolors="k",
                       lw=0.6, zorder=5)                                                # end x_0
        ax.set_title(title, fontsize=11)
        style_axes(ax)
    # shared legend explaining the markers (colors = trajectory identity)
    start_h = Line2D([], [], color="0.3", marker="x", ls="none", ms=8, mew=2, label="start $x_T$")
    end_h = Line2D([], [], color="0.3", marker="o", ls="none", ms=8, mec="k", label="end $x_0$")
    origin_h = Line2D([], [], color="gold", marker="*", ls="none", ms=12, mec="k",
                      label="origin (data mean)")
    axes[1].legend(handles=[start_h, end_h, origin_h], loc="upper right", fontsize=9)
    fig.suptitle(f"{n_show} reverse trajectories on the Swiss roll — effect of the noise-injection term",
                 y=1.02, fontsize=12)
    fig.savefig(f"{FIG}/spiral_traj_noise.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved figures/spiral_traj_noise.png")


def fig_vary_T(data, T_values=(10, 50, 200, 1000)):
    fig, axes = plt.subplots(1, len(T_values), figsize=(4.2 * len(T_values), 4.4))
    for ax, T in zip(axes, T_values):
        print(f"=== training spiral DDPM at T={T} ===")
        model, diffusion = train(T, data)
        traj = sample_trajectory(model, diffusion, n=2000, add_noise=True, seed=2)
        draw_trajectories(ax, traj, data, f"T = {T}")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Reverse-process trajectories on the Swiss roll — effect of number of steps T",
                 y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/spiral_traj_T.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved figures/spiral_traj_T.png")


def main():
    data = make_spiral(4000)
    print("=== training main spiral DDPM (T=1000) ===")
    model, diffusion = train(1000, data)
    fig_samples(model, diffusion, data)
    fig_noise_on_off(model, diffusion, data)
    fig_vary_T(data)
    print("ALL DONE")


if __name__ == "__main__":
    main()
