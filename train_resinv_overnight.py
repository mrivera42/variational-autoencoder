"""Overnight: does BAND-LIMITED (colored) noise make FNO-DDPM sampling actually
resolution-invariant? Train two FNO-DDPMs on MNIST 28x28 -- one with white noise,
one with band-limited noise -- then sample at 28/40/56 and score sample quality
with a small MNIST classifier (recognizability + digit diversity).

Band-limited noise uses a FIXED physical frequency cutoff (cycles-per-image), so
it's the same smooth field at any resolution -> the DDO/inf-Diff idea, minimal form.

    uv run python train_resinv_overnight.py
Env: EPOCHS (18), CUTOFF (8)
"""

import os, json, traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from diffusion import GaussianDiffusion
from fno_ddpm import FNODenoiser

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FIG, CKPT = "figures", "checkpoints"
os.makedirs(FIG, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
T = 1000
EPOCHS = int(os.environ.get("EPOCHS", 18))
CUTOFF = float(os.environ.get("CUTOFF", 8))          # cycles-per-image low-pass cutoff
EVAL_RES = [28, 40, 56]
RESULTS = {}
_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# ---------- noise ----------
def band_limited_noise(shape, device):
    B, C, H, W = shape
    white = torch.randn(B, C, H, W, device=device)
    fh = torch.fft.fftfreq(H, device=device) * H          # cycles per image
    fw = torch.fft.rfftfreq(W, device=device) * W
    r = torch.sqrt(fh[:, None] ** 2 + fw[None, :] ** 2)    # [H, Wf] absolute radial freq
    env = torch.exp(-(r / CUTOFF) ** 2)                    # fixed physical low-pass
    n = torch.fft.irfft2(torch.fft.rfft2(white) * env, s=(H, W))
    return n / (n.std(dim=(1, 2, 3), keepdim=True) + 1e-8)  # unit per-sample variance


def noise_fn(mode, shape):
    return torch.randn(shape, device=DEVICE) if mode == "white" else band_limited_noise(shape, DEVICE)


# ---------- classifier (sample-quality metric) ----------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10))                      # fixed 28-in

    def forward(self, x):
        return self.net(x)


def train_classifier():
    path = f"{CKPT}/mnist_classifier.pth"
    clf = SmallCNN().to(DEVICE)
    if os.path.exists(path):
        clf.load_state_dict(torch.load(path, map_location=DEVICE)); clf.eval(); return clf
    dl = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=_TF),
                    batch_size=256, shuffle=True)
    opt = optim.Adam(clf.parameters(), 1e-3)
    for _ in range(2):
        for x, y in tqdm(dl, desc="classifier", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval(); torch.save(clf.state_dict(), path)
    return clf


@torch.no_grad()
def score_samples(clf, imgs):                              # imgs: [n,H,W] in [0,1]
    x = torch.tensor(imgs, device=DEVICE).unsqueeze(1) * 2 - 1
    if x.shape[-1] != 28:
        x = F.interpolate(x, size=28, mode="bilinear", align_corners=False)
    p = F.softmax(clf(x), dim=1)
    recog = p.max(1).values.mean().item()                  # confidence it's a clear digit
    cls_hist = p.mean(0)                                    # avg class distribution
    diversity = (-(cls_hist * (cls_hist + 1e-9).log()).sum() / np.log(10)).item()  # 0..1
    return recog, diversity


# ---------- diffusion train / sample ----------
def train(model, diffusion, mode, epochs):
    dl = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=_TF),
                    batch_size=128, shuffle=True)
    opt = optim.Adam(model.parameters(), 2e-4)
    losses = []
    for ep in range(epochs):
        run = 0.0
        for x, _ in tqdm(dl, desc=f"{mode} ep{ep}", leave=False):
            x = x.to(DEVICE)
            t = torch.randint(0, T, (x.size(0),), device=DEVICE)
            noise = noise_fn(mode, x.shape)
            loss = diffusion.p_losses(model, x, t, noise=noise)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * x.size(0)
        losses.append(run / len(dl.dataset))
        print(f"[{mode}] epoch {ep} loss {losses[-1]:.4f}", flush=True)
    torch.save(model.state_dict(), f"{CKPT}/ddpm_fno_{mode}.pth")
    return losses


@torch.no_grad()
def sample(model, diffusion, res, mode, n=8):
    model.eval()
    x = noise_fn(mode, (n, 1, res, res))
    for step in tqdm(range(T - 1, -1, -1), desc=f"sample {mode}@{res}", leave=False):
        t = torch.full((n,), step, device=DEVICE, dtype=torch.long)
        eps = model(x, t)
        abar_t, beta_t = diffusion.alpha_bars[step], diffusion.betas[step]
        alpha_t = 1 - beta_t
        x0 = ((x - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)).clamp(-1, 1)
        abar_prev = diffusion.alpha_bars[step - 1] if step > 0 else torch.ones_like(abar_t)
        mean = (beta_t * torch.sqrt(abar_prev) / (1 - abar_t)) * x0 \
             + ((1 - abar_prev) * torch.sqrt(alpha_t) / (1 - abar_t)) * x
        if step > 0:
            var = beta_t * (1 - abar_prev) / (1 - abar_t)
            x = mean + torch.sqrt(var) * noise_fn(mode, x.shape)
        else:
            x = mean
    model.train()
    return ((x.clamp(-1, 1) + 1) / 2).view(n, res, res).cpu().numpy()


def grid(rows, title, path, n=8):
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.4, len(rows) * 1.5))
    for r, (lab, imgs) in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(imgs[i], cmap="gray"); axes[r, i].axis("off")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(lab, fontsize=10)
    fig.suptitle(title, y=1.0); fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", path, flush=True)


def run_mode(mode, clf):
    diffusion = GaussianDiffusion(T, device=DEVICE)
    model = FNODenoiser(width=48, n_blocks=4, im_shape=(28, 28), ksize=15).to(DEVICE)
    losses = train(model, diffusion, mode, EPOCHS)
    rows, metrics = [], {}
    for res in EVAL_RES:
        imgs = sample(model, diffusion, res, mode)
        recog, div = score_samples(clf, imgs)
        metrics[res] = {"recognizability": recog, "diversity": div}
        rows.append((f"{res}x{res}", imgs))
        print(f"[metric] {mode}@{res}: recog={recog:.3f} diversity={div:.3f}", flush=True)
    RESULTS[mode] = {"final_loss": losses[-1], "metrics": metrics}
    grid(rows, f"FNO-DDPM ({mode} noise): trained at 28, sampled at 28/40/56",
         f"{FIG}/resinv_{mode}_multires.png")
    with open(f"{FIG}/resinv_metrics.json", "w") as f:
        json.dump(RESULTS, f, indent=2)


def summary_plot():
    if not RESULTS:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for mode, d in RESULTS.items():
        rs = sorted(d["metrics"])
        ax.plot(rs, [d["metrics"][r]["recognizability"] for r in rs], marker="o", label=f"{mode} noise")
    ax.set_xlabel("sampling resolution"); ax.set_ylabel("classifier recognizability (higher=better)")
    ax.set_title("Sample quality vs resolution: white vs band-limited noise")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/resinv_quality_metric.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved figures/resinv_quality_metric.png", flush=True)


if __name__ == "__main__":
    clf = train_classifier()
    for mode in ["white", "colored"]:
        print(f"\n===== {mode} =====", flush=True)
        try:
            run_mode(mode, clf)
        except Exception:
            print(f"[{mode}] FAILED:\n{traceback.format_exc()}", flush=True)
    summary_plot()
    print("\nRESULTS:", json.dumps(RESULTS, indent=2), flush=True)
    print("ALL DONE", flush=True)
