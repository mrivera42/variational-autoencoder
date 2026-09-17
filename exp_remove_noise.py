"""Experiment 1 (advisor bullet 1): what does the noise-injection term do?

DDPM reverse step:   x_{t-1} = mean(x_t, eps_hat) + sigma_t * z
We generate from a trained model TWICE, starting from the SAME initial noise:
  (a) standard DDPM               -> keep the sigma_t * z term
  (b) noise term commented out    -> drop it (reverse step = mean only)
and compare the samples side by side.

This changes only sampling (no retraining). We run it on the latent model (fast)
and on the already-trained pixel model. Uses diffusion.sample(..., add_noise=False)
to drop the term.

    uv run python exp_remove_noise.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from inference_ddpm import load_ddpm
from train_ddpm import load_vae

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
N = 8       # samples per row
SEED = 0    # fixes x_T so the two settings are compared on the same starting noise


@torch.no_grad()
def generate(run_name, add_noise):
    model, diffusion, cfg = load_ddpm(run_name)
    torch.manual_seed(SEED)   # same initial x_T for add_noise True vs False
    if cfg["space"] == "pixel":
        x0 = diffusion.sample(model, (N, 1, 28, 28), add_noise=add_noise)
        return ((x0.clamp(-1, 1) + 1) / 2).view(N, 28, 28).cpu().numpy()
    vae, _ = load_vae(cfg["vae_name"])
    z0 = diffusion.sample(model, (N, cfg["latent_dim"]), add_noise=add_noise)
    return vae.decode(z0).view(N, 28, 28).cpu().numpy()


def make_figure(run_name, title, path):
    rows = [
        ("DDPM (with noise term)", generate(run_name, add_noise=True)),
        ("noise term removed", generate(run_name, add_noise=False)),
    ]
    fig, axes = plt.subplots(2, N, figsize=(N * 1.3, 3.2))
    for r, (label, imgs) in enumerate(rows):
        for i in range(N):
            axes[r, i].imshow(imgs[i], cmap="gray")
            axes[r, i].axis("off")
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    make_figure("ddpm_latent_z20",
                "Removing the noise-injection term during generation — Latent DDPM",
                f"{FIG}/exp1_remove_noise_latent.png")
    make_figure("ddpm_pixel",
                "Removing the noise-injection term during generation — Pixel DDPM",
                f"{FIG}/exp1_remove_noise_pixel.png")
