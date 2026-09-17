"""Train a neural-operator DDPM (FNODenoiser) on MNIST at 28x28, then sample at
MULTIPLE resolutions from the SAME model -- the train-one-resolution /
sample-another demonstration a standard UNet-DDPM cannot do.

    uv run python train_fno_ddpm.py
Env overrides: EPOCHS (default 20).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from diffusion import GaussianDiffusion
from fno_ddpm import FNODenoiser

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FIG, CKPT = "figures", "checkpoints"
os.makedirs(FIG, exist_ok=True); os.makedirs(CKPT, exist_ok=True)

EPOCHS = int(os.environ.get("EPOCHS", 20))
BATCH, LR, T = 128, 2e-4, 1000
SAMPLE_RES = [28, 40, 56]        # native, then two unseen resolutions


def loaders():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # [-1,1]
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    return DataLoader(tr, batch_size=BATCH, shuffle=True)


@torch.no_grad()
def sample_grid(model, diffusion, res, n=8):
    model.eval()
    x0 = diffusion.sample(model, (n, 1, res, res), progress=True)
    model.train()
    return ((x0.clamp(-1, 1) + 1) / 2).view(n, res, res).cpu().numpy()


def save_multires(model, diffusion, path, tag=""):
    fig, axes = plt.subplots(len(SAMPLE_RES), 8, figsize=(8 * 1.5, len(SAMPLE_RES) * 1.6))
    for r, res in enumerate(SAMPLE_RES):
        imgs = sample_grid(model, diffusion, res)
        for i in range(8):
            axes[r, i].imshow(imgs[i], cmap="gray"); axes[r, i].axis("off")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"{res}x{res}", fontsize=11)
    fig.suptitle(f"FNO-DDPM trained at 28x28, sampled at multiple resolutions {tag}", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", path)


def main():
    train_loader = loaders()
    diffusion = GaussianDiffusion(num_timesteps=T, beta_start=1e-4, beta_end=0.02, device=DEVICE)
    model = FNODenoiser(width=48, n_blocks=4, im_shape=(28, 28), ksize=15).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    print(f"device={DEVICE}  params={sum(p.numel() for p in model.parameters()):,}  epochs={EPOCHS}")

    losses = []
    for epoch in range(EPOCHS):
        running = 0.0
        for x, _ in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x = x.to(DEVICE)
            t = torch.randint(0, T, (x.size(0),), device=DEVICE)
            loss = diffusion.p_losses(model, x, t)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * x.size(0)
        ep = running / len(train_loader.dataset)
        losses.append(ep)
        print(f"[fno_ddpm] epoch {epoch}, loss {ep:.4f}")
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            torch.save(model.state_dict(), f"{CKPT}/ddpm_fno.pth")
            save_multires(model, diffusion, f"{FIG}/fno_ddpm_multires.png", tag=f"(epoch {epoch})")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("noise-prediction MSE"); ax.grid(alpha=0.3)
    ax.set_title("FNO-DDPM training loss")
    fig.tight_layout(); fig.savefig(f"{FIG}/fno_ddpm_loss.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("ALL DONE")


if __name__ == "__main__":
    main()
