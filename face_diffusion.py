"""Train a pixel DDPM on a very small face dataset.

Dataset: Olivetti faces (400 grayscale 64x64 images, 40 people) — tiny, built into
scikit-learn. Resized to 32x32 and normalized to [-1, 1]. Reuses the exact same
UNet + GaussianDiffusion as the MNIST pixel DDPM (no new machinery).

Saves intermediate sample grids + checkpoint every few thousand iters, so there
are artifacts even if you stop it early. Logs to W&B project "ddpm-faces".

    uv run python face_diffusion.py

Caveat: 400 images is very small, so the model will partly *memorize* the training
faces rather than generalize — expected, and worth a nearest-neighbour check later.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_olivetti_faces
from tqdm import tqdm
import wandb

from diffusion import GaussianDiffusion
from ddpm_model import UNet
from train_ddpm import DEVICE

FIG = "figures"
CKPT = "checkpoints"
os.makedirs(FIG, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

RES = 32
T = 1000
ITERS = 8000
BATCH = 64
LR = 2e-4


def load_faces():
    data = fetch_olivetti_faces()
    imgs = torch.tensor(data.images, dtype=torch.float32)[:, None]     # [400,1,64,64] in [0,1]
    imgs = F.interpolate(imgs, size=RES, mode="bilinear", align_corners=False)
    return imgs * 2 - 1                                                # -> [-1,1]


@torch.no_grad()
def save_samples(model, diffusion, path, title, n=16):
    model.eval()
    x0 = diffusion.sample(model, (n, 1, RES, RES))
    imgs = ((x0.clamp(-1, 1) + 1) / 2).cpu().numpy()
    cols, rows = 4, n // 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i, 0], cmap="gray")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    model.train()
    print("saved", path)


def save_real_grid(data, path, n=16):
    idx = torch.randperm(data.shape[0])[:n]
    imgs = ((data[idx].clamp(-1, 1) + 1) / 2).cpu().numpy()
    cols, rows = 4, n // 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i, 0], cmap="gray")
        ax.axis("off")
    fig.suptitle("Olivetti faces — real (32x32)")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    wandb.init(project="ddpm-faces", name="olivetti_pixel_32",
               config={"res": RES, "T": T, "iters": ITERS, "batch": BATCH,
                       "lr": LR, "device": DEVICE, "dataset": "olivetti_400"},
               reinit=True)

    data = load_faces().to(DEVICE)
    print(f"faces: {tuple(data.shape)}  device {DEVICE}")
    save_real_grid(data, f"{FIG}/faces_real.png")

    diffusion = GaussianDiffusion(num_timesteps=T, beta_start=1e-4, beta_end=0.02, device=DEVICE)
    model = UNet(in_ch=1, base_ch=64, time_dim=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    N = data.shape[0]
    losses = []
    for it in tqdm(range(ITERS), desc="train"):
        idx = torch.randint(0, N, (BATCH,), device=DEVICE)
        x0 = data[idx]
        t = torch.randint(0, T, (BATCH,), device=DEVICE)
        loss = diffusion.p_losses(model, x0, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if (it + 1) % 200 == 0:
            wandb.log({"iter": it + 1, "loss": float(np.mean(losses[-200:]))})
        if (it + 1) % 2000 == 0:
            save_samples(model, diffusion, f"{FIG}/faces_samples_it{it+1}.png",
                         f"Olivetti DDPM — {it+1} iters")
            torch.save(model.state_dict(), f"{CKPT}/ddpm_faces.pth")

    save_samples(model, diffusion, f"{FIG}/faces_samples.png", "Olivetti DDPM samples (final)")
    torch.save(model.state_dict(), f"{CKPT}/ddpm_faces.pth")

    # loss curve (raw + smoothed)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(losses, alpha=0.3, lw=0.7, label="per-iter")
    k = 100
    sm = np.convolve(losses, np.ones(k) / k, mode="valid")
    ax.plot(range(k - 1, len(losses)), sm, color="crimson", lw=2, label=f"smoothed ({k})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("noise-prediction MSE")
    ax.set_title("Olivetti face DDPM — training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG}/faces_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/faces_loss.png")

    wandb.finish()
    print("ALL DONE")


if __name__ == "__main__":
    main()
