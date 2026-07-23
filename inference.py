"""Load a trained VAE checkpoint and visualize:
  1. Samples from the prior (z ~ N(0, I) -> decode)
  2. Reconstructions of held-out test images

Usage: uv run python inference.py [run_name]
       defaults to "baseline_z20"
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import VAE
from train import CHECKPOINT_DIR, DEVICE


FIGURES_DIR = "figures"


def load_model(run_name):
    cfg_path = os.path.join(CHECKPOINT_DIR, f"{run_name}.json")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_name}.pth")
    with open(cfg_path) as f:
        config = json.load(f)
    model = VAE(
        in_dim=config["in_dim"],
        trunk_dim=config["trunk_dim"],
        latent_dim=config["latent_dim"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model, config


def plot_samples(model, config, n=16, save_path=None):
    """Sample z ~ N(0, I), decode, plot as a grid."""
    with torch.no_grad():
        z = torch.randn(n, config["latent_dim"], device=DEVICE)
        imgs = model.decode(z).view(n, 28, 28).cpu().numpy()

    cols = 4
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(f"Generations from N(0, I) — {config['name']}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  saved {save_path}")
    plt.close(fig)


def plot_reconstructions(model, config, n=8, save_path=None):
    """Reconstruct held-out test images, plot input vs recon side-by-side."""
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(test_dataset, batch_size=n, shuffle=True)
    x, _ = next(iter(loader))
    x = x.to(DEVICE).view(n, -1)

    with torch.no_grad():
        recon, _, _ = model(x)

    x = x.view(n, 28, 28).cpu().numpy()
    recon = recon.view(n, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(x[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("input", loc="left")
    axes[1, 0].set_title("recon", loc="left")
    fig.suptitle(f"Reconstructions — {config['name']}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  saved {save_path}")
    plt.close(fig)


def visualize(run_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model, config = load_model(run_name)
    plot_samples(model, config, save_path=os.path.join(FIGURES_DIR, f"{run_name}_samples.png"))
    plot_reconstructions(model, config, save_path=os.path.join(FIGURES_DIR, f"{run_name}_recons.png"))


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "baseline_z20"
    visualize(run_name)