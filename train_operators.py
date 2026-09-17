"""Overnight: train three neural-operator DDPM denoisers on MNIST (28x28), show
resolution generalization QUALITATIVELY (sample at 28/40/56) and QUANTITATIVELY
(noise-prediction MSE on the TEST set resized to each resolution).

  1. CNN  -> convert to spectral at inference (user's kernel_to_spectral method)
  2. FNO  -> spectral weights trained from scratch
  3. GNO  -> graph-neural-operator denoiser (bug-fixed layer)

Quant metric: for each model, ε-MSE on resized test digits at res in {28,40,56}.
Resolution-invariant operators stay low; the naive CNN spikes. (White noise limits
invariance for all -> operator-vs-naive-CNN is the informative comparison.)

Each model wrapped in try/except; figures + checkpoints saved as each finishes.

    uv run python train_operators.py
Env: CNN_EPOCHS(15) FNO_EPOCHS(12) GNO_EPOCHS(8)
"""

import os, json, traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from diffusion import GaussianDiffusion

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FIG, CKPT = "figures", "checkpoints"
os.makedirs(FIG, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
T = 1000
EVAL_RES = [28, 40, 56]
RESULTS = {}                       # model_name -> {res: eps_mse}
_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def train_loader(batch):
    ds = datasets.MNIST("./data", train=True, download=True, transform=_TF)
    return DataLoader(ds, batch_size=batch, shuffle=True)


def train(model, diffusion, epochs, batch, tag):
    dl = train_loader(batch)
    opt = optim.Adam(model.parameters(), lr=2e-4)
    losses = []
    for ep in range(epochs):
        run = 0.0
        for x, _ in tqdm(dl, desc=f"{tag} ep{ep}", leave=False):
            x = x.to(DEVICE)
            t = torch.randint(0, T, (x.size(0),), device=DEVICE)
            loss = diffusion.p_losses(model, x, t)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * x.size(0)
        losses.append(run / len(dl.dataset))
        print(f"[{tag}] epoch {ep} loss {losses[-1]:.4f}", flush=True)
    torch.save(model.state_dict(), f"{CKPT}/ddpm_{tag}.pth")
    return losses


@torch.no_grad()
def eval_eps_mse(model, diffusion, res, n_batches=20, batch=64):
    """Mean noise-prediction MSE on the TEST set resized to `res` (fixed seed)."""
    model.eval()
    ds = datasets.MNIST("./data", train=False, download=True, transform=_TF)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    tot, n = 0.0, 0
    for i, (x, _) in enumerate(dl):
        if i >= n_batches:
            break
        x = x.to(DEVICE)
        if res != 28:
            x = F.interpolate(x, size=res, mode="bilinear", align_corners=False)
        t = torch.randint(0, T, (x.size(0),), device=DEVICE, generator=g)
        noise = torch.randn(x.shape, device=DEVICE, generator=g)
        loss = diffusion.p_losses(model, x, t, noise=noise)
        tot += loss.item() * x.size(0); n += x.size(0)
    model.train()
    return tot / n


def evaluate(model, diffusion, name, res_list=EVAL_RES):
    RESULTS[name] = {r: eval_eps_mse(model, diffusion, r) for r in res_list}
    print(f"[eval] {name}: " + "  ".join(f"{r}:{RESULTS[name][r]:.4f}" for r in res_list), flush=True)
    with open(f"{FIG}/operator_metrics.json", "w") as f:
        json.dump(RESULTS, f, indent=2)


@torch.no_grad()
def sample(model, diffusion, res, n=8):
    model.eval()
    x0 = diffusion.sample(model, (n, 1, res, res), progress=True)
    model.train()
    return ((x0.clamp(-1, 1) + 1) / 2).view(n, res, res).cpu().numpy()


def grid(rows, title, path, n=8):
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.4, len(rows) * 1.5))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (label, imgs) in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(imgs[i], cmap="gray"); axes[r, i].axis("off")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=10)
    fig.suptitle(title, y=1.0)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", path, flush=True)


def loss_plot(losses, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("noise-prediction MSE"); ax.grid(alpha=0.3)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def metric_plot():
    if not RESULTS:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for name, d in RESULTS.items():
        xs = sorted(d); ax.plot(xs, [d[r] for r in xs], marker="o", label=name)
    ax.axhline(1.0, ls="--", c="0.6", lw=1, label="predict-zero baseline")
    ax.set_xlabel("sampling / evaluation resolution")
    ax.set_ylabel("noise-prediction MSE on resized test set")
    ax.set_title("Resolution generalization of the denoiser (lower = better)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/operator_resolution_metric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/operator_resolution_metric.png", flush=True)


def run_cnn():
    from cnn_ddpm import CNNDenoiser, convert_to_spectral
    diffusion = GaussianDiffusion(T, device=DEVICE)
    model = CNNDenoiser(width=48, n_blocks=4, ksize=7).to(DEVICE)
    print(f"[cnn] params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    losses = train(model, diffusion, int(os.environ.get("CNN_EPOCHS", 15)), 128, "cnn")
    loss_plot(losses, "CNN-DDPM training loss", f"{FIG}/cnn_ddpm_loss.png")
    spec = convert_to_spectral(model, (28, 28)).to(DEVICE)      # user's conversion method
    evaluate(model, diffusion, "CNN (naive)")
    evaluate(spec, diffusion, "CNN->spectral")
    rows = [
        ("CNN @28",         sample(model, diffusion, 28)),
        ("->spectral @28",  sample(spec,  diffusion, 28)),
        ("->spectral @56",  sample(spec,  diffusion, 56)),
        ("CNN @56 (naive)", sample(model, diffusion, 56)),
    ]
    grid(rows, "CNN-DDPM: convert to spectral (Fourier) and sample at new resolutions",
         f"{FIG}/cnn_convert_multires.png")
    metric_plot()


def run_fno():
    from fno_ddpm import FNODenoiser
    diffusion = GaussianDiffusion(T, device=DEVICE)
    model = FNODenoiser(width=48, n_blocks=4, im_shape=(28, 28), ksize=15).to(DEVICE)
    print(f"[fno] params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    losses = train(model, diffusion, int(os.environ.get("FNO_EPOCHS", 12)), 128, "fno")
    loss_plot(losses, "FNO-DDPM (from scratch) training loss", f"{FIG}/fno_ddpm_loss.png")
    evaluate(model, diffusion, "FNO (scratch)")
    rows = [(f"{res}x{res}", sample(model, diffusion, res)) for res in (28, 40, 56)]
    grid(rows, "FNO-DDPM (from scratch): trained at 28, sampled at multiple resolutions",
         f"{FIG}/fno_ddpm_multires.png")
    metric_plot()


def run_gno():
    from gno_ddpm import GNODenoiser
    diffusion = GaussianDiffusion(T, device=DEVICE)
    model = GNODenoiser(radius=0.1, width=24, n_blocks=3).to(DEVICE)
    print(f"[gno] params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    losses = train(model, diffusion, int(os.environ.get("GNO_EPOCHS", 8)), 32, "gno")
    loss_plot(losses, "GNO-DDPM training loss", f"{FIG}/gno_ddpm_loss.png")
    evaluate(model, diffusion, "GNO", res_list=[28, 56])
    rows = [(f"{res}x{res}", sample(model, diffusion, res)) for res in (28, 56)]
    grid(rows, "GNO-DDPM: trained at 28, sampled at multiple resolutions",
         f"{FIG}/gno_ddpm_multires.png")
    metric_plot()


if __name__ == "__main__":
    for name, fn in [("CNN", run_cnn), ("FNO", run_fno), ("GNO", run_gno)]:
        print(f"\n===== {name} =====", flush=True)
        try:
            fn()
        except Exception:
            print(f"[{name}] FAILED:\n{traceback.format_exc()}", flush=True)
    metric_plot()
    print("\nRESULTS:", json.dumps(RESULTS, indent=2), flush=True)
    print("ALL DONE", flush=True)
