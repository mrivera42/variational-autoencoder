"""Load a trained DDPM checkpoint and visualize samples.

  pixel  : sample x_0 in [-1, 1], map back to [0, 1], plot as a grid.
  latent : sample a latent z_0, decode it with the frozen VAE, plot as a grid.

The plotting/loading is done; it relies on GaussianDiffusion.sample (which you
implement in diffusion.py). Parallels the VAE's inference.py.

Usage: uv run python inference_ddpm.py [run_name]
       defaults to "ddpm_pixel"
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import torch

from diffusion import GaussianDiffusion
from train_ddpm import CHECKPOINT_DIR, DEVICE, build_model, load_vae


FIGURES_DIR = "figures"


def load_ddpm(run_name):
    """Return (model, diffusion, config) for a trained DDPM run."""
    cfg_path = os.path.join(CHECKPOINT_DIR, f"{run_name}.json")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_name}.pth")
    with open(cfg_path) as f:
        config = json.load(f)

    model = build_model(config)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=config["num_timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        device=DEVICE,
    )
    return model, diffusion, config


@torch.no_grad()
def sample_images(model, diffusion, config, n=16):
    """Draw n samples and return them as [n, 28, 28] in [0, 1] for display."""
    if config["space"] == "pixel":
        x0 = diffusion.sample(model, shape=(n, 1, 28, 28))    # in [-1, 1]
        imgs = (x0.clamp(-1, 1) + 1) / 2                       # -> [0, 1]
        return imgs.view(n, 28, 28).cpu().numpy()

    # latent: sample in latent space, then decode with the frozen VAE
    vae, _ = load_vae(config["vae_name"])
    z0 = diffusion.sample(model, shape=(n, config["latent_dim"]))
    imgs = vae.decode(z0)                                      # [n, 784] in [0, 1]
    return imgs.view(n, 28, 28).cpu().numpy()


def plot_grid(imgs, title, save_path=None):
    n = imgs.shape[0]
    cols = 4
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  saved {save_path}")
    plt.close(fig)


def visualize(run_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model, diffusion, config = load_ddpm(run_name)
    imgs = sample_images(model, diffusion, config)
    plot_grid(imgs, f"DDPM samples — {config['name']} ({config['space']})",
              save_path=os.path.join(FIGURES_DIR, f"{run_name}_samples.png"))


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "ddpm_pixel"
    visualize(run_name)
