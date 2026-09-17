"""Multiple-trials grid: scaling the VAE latent draw (the VAE analog of the DDPM
lambda sweep). Each row is a scenario, each column an independent trial.

  z = 0        -> deterministic prior mean; every trial IDENTICAL (collapse)
  z ~ N(0,.5I) -> reduced-variance draw; low-diversity, blobby digits
  z ~ N(0, I)  -> full prior; diverse, specific digits

    uv run python vae_collapse_grid.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import VAE

CKPT, OUT, NAME = "checkpoints", "figures", "baseline_z20"
os.makedirs(OUT, exist_ok=True)

cfg = json.load(open(f"{CKPT}/{NAME}.json"))
vae = VAE(in_dim=cfg["in_dim"], trunk_dim=cfg["trunk_dim"], latent_dim=cfg["latent_dim"])
vae.load_state_dict(torch.load(f"{CKPT}/{NAME}.pth", map_location="cpu"))
vae.eval()
D = cfg["latent_dim"]

NCOL = 8
SCENARIOS = [("z = 0\n(prior mean)", 0.0), ("z ~ N(0, ½I)", 0.5), ("z ~ N(0, I)", 1.0)]

torch.manual_seed(0)
fig, axes = plt.subplots(len(SCENARIOS), NCOL, figsize=(NCOL * 1.3, len(SCENARIOS) * 1.5))
with torch.no_grad():
    for r, (label, std) in enumerate(SCENARIOS):
        z = torch.zeros(NCOL, D) if std == 0.0 else std * torch.randn(NCOL, D)
        imgs = vae.decode(z).view(NCOL, 28, 28).numpy()
        for c in range(NCOL):
            axes[r, c].imshow(imgs[c], cmap="gray")
            axes[r, c].axis("off")
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=10)

fig.suptitle("VAE decode: prior mean (z = 0) vs sampled latents — 8 trials per row",
             y=1.0, fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/vae_collapse_grid.png", dpi=150, bbox_inches="tight")
print("saved figures/vae_collapse_grid.png")
