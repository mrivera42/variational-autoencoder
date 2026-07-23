"""Overnight pipeline: train pixel DDPM + latent DDPM, save checkpoints, and
produce labeled slide figures. Run from the variational-autoencoder repo dir.

Figures written to figures/:
  ddpm_pixel_loss.png      ddpm_pixel_samples.png
  ddpm_latent_loss.png     ddpm_latent_samples.png
  ddpm_comparison.png
"""

import os
# wandb logs by default (project "ddpm-mnist"). To run without it:
#   WANDB_MODE=disabled uv run python make_figures.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_ddpm import run_experiment, load_vae, DEVICE
from inference_ddpm import load_ddpm

FIG = "figures"
os.makedirs(FIG, exist_ok=True)

# Overridable via env for a quick smoke test, e.g.:
#   PIXEL_EPOCHS=1 LATENT_EPOCHS=1 N_SAMPLES=4 python run_overnight.py
PIXEL_EPOCHS = int(os.environ.get("PIXEL_EPOCHS", 20))
LATENT_EPOCHS = int(os.environ.get("LATENT_EPOCHS", 60))
N_SAMPLES = int(os.environ.get("N_SAMPLES", 8))

# ---- configs (epochs tuned to finish overnight) ---------------------------
PIXEL_CFG = {
    "name": "ddpm_pixel", "space": "pixel",
    "batch_size": 128, "lr": 2e-4, "num_epochs": PIXEL_EPOCHS,
    "num_timesteps": 1000, "beta_start": 1e-4, "beta_end": 0.02,
    "time_dim": 128, "base_ch": 64,
}
LATENT_CFG = {
    "name": "ddpm_latent_z20", "space": "latent", "vae_name": "baseline_z20",
    "batch_size": 128, "lr": 1e-3, "num_epochs": LATENT_EPOCHS,
    "num_timesteps": 1000, "beta_start": 1e-4, "beta_end": 0.02,
    "time_dim": 128, "hidden_dim": 256,
}


def real_images(n):
    """n real MNIST test images as [n, 28, 28] in [0, 1]."""
    ds = datasets.MNIST(root="./data", train=False, download=True,
                        transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=n, shuffle=True)
    x, _ = next(iter(loader))
    return x.view(n, 28, 28).numpy()


@torch.no_grad()
def sample_grid(run_name, n):
    """Return n generated images as [n, 28, 28] in [0, 1]."""
    model, diffusion, cfg = load_ddpm(run_name)
    if cfg["space"] == "pixel":
        x0 = diffusion.sample(model, (n, 1, 28, 28), progress=True)
        imgs = ((x0.clamp(-1, 1) + 1) / 2).view(n, 28, 28)
    else:
        vae, _ = load_vae(cfg["vae_name"])
        z0 = diffusion.sample(model, (n, cfg["latent_dim"]), progress=True)
        imgs = vae.decode(z0).view(n, 28, 28)
    return imgs.cpu().numpy()


def plot_loss(res, title, path):
    epochs = range(1, len(res["train_losses"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, res["train_losses"], marker="o", ms=3, label="train")
    ax.plot(epochs, res["test_losses"], marker="s", ms=3, label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise-prediction MSE loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_real_vs_gen(real, gen, gen_label, title, path):
    n = real.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3.0))
    for i in range(n):
        axes[0, i].imshow(real[i], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(gen[i], cmap="gray");  axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Real", rotation=0, ha="right", va="center", fontsize=11)
    axes[1, 0].set_ylabel(gen_label, rotation=0, ha="right", va="center", fontsize=11)
    # re-enable the y-axis spine just enough to show the row labels
    for row, lab in [(0, "Real MNIST"), (1, gen_label)]:
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(lab, rotation=90, fontsize=10)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_comparison(real, pix, lat, path):
    n = real.shape[0]
    rows = [("Real MNIST", real), ("Pixel DDPM", pix), ("Latent DDPM", lat)]
    fig, axes = plt.subplots(3, n, figsize=(n * 1.3, 4.4))
    for r, (lab, imgs) in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(imgs[i], cmap="gray"); axes[r, i].axis("off")
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(lab, rotation=90, fontsize=10)
    fig.suptitle("MNIST: real vs. pixel-space DDPM vs. latent DDPM samples", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    print("=== training pixel DDPM ===")
    res_pix = run_experiment(dict(PIXEL_CFG))
    print("=== training latent DDPM ===")
    res_lat = run_experiment(dict(LATENT_CFG))

    plot_loss(res_pix, "Pixel-space DDPM — training curve", f"{FIG}/ddpm_pixel_loss.png")
    plot_loss(res_lat, "Latent DDPM (z=20) — training curve", f"{FIG}/ddpm_latent_loss.png")

    print("=== sampling ===")
    n = N_SAMPLES
    real = real_images(n)
    pix = sample_grid("ddpm_pixel", n)
    lat = sample_grid("ddpm_latent_z20", n)

    plot_real_vs_gen(real, pix, "Pixel DDPM",
                    "Pixel-space DDPM: real vs. generated", f"{FIG}/ddpm_pixel_samples.png")
    plot_real_vs_gen(real, lat, "Latent DDPM",
                    "Latent DDPM (z=20): real vs. generated", f"{FIG}/ddpm_latent_samples.png")
    plot_comparison(real, pix, lat, f"{FIG}/ddpm_comparison.png")

    print("ALL DONE")


if __name__ == "__main__":
    main()
