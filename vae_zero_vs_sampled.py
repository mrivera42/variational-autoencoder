"""Test the 'decode z=0 -> the average / collapse' claim on the trained MNIST VAE.

Compares:
  - decode(z = 0)            : the prior MEAN -> should be a single blurry "average digit"
  - mean over many decode(z) : the empirical average -> also a blurry mean digit
  - decode(z ~ N(0, I))      : specific sampled digits -> varied, recognizable
This is the VAE analog of the DDPM lambda=0 collapse: skip the random draw and you
get the dataset mean; sample z first and you get a specific (blurry) digit.

    uv run python vae_zero_vs_sampled.py
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

torch.manual_seed(0)
with torch.no_grad():
    img0 = vae.decode(torch.zeros(1, D)).view(28, 28).numpy()          # z = 0
    decs = vae.decode(torch.randn(4000, D))                            # z ~ N(0, I)
    avg = decs.mean(0).view(28, 28).numpy()                            # empirical average
    samp = decs[:6].view(6, 28, 28).numpy()                            # individual samples

fig, axes = plt.subplots(1, 8, figsize=(12, 2.1))
axes[0].imshow(img0, cmap="gray"); axes[0].set_title("z = 0\n(prior mean)", fontsize=9)
axes[1].imshow(avg, cmap="gray");  axes[1].set_title("avg of\n4000 decodes", fontsize=9)
for i in range(6):
    axes[2 + i].imshow(samp[i], cmap="gray")
    axes[2 + i].set_title("z ~ N(0, I)  (sampled) →" if i == 0 else "", fontsize=9, loc="left")
for a in axes:
    a.axis("off")
fig.suptitle("VAE decode: z = 0  vs  sampled z ~ N(0, I)", y=1.08)
fig.tight_layout()
fig.savefig(f"{OUT}/vae_zero_vs_sampled.png", dpi=150, bbox_inches="tight")
print("saved figures/vae_zero_vs_sampled.png")
