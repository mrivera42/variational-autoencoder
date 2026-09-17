"""Zoomed MSE + diversity vs resolution, white vs colored (band-limited) noise.
Each model is evaluated with ITS OWN noise type (fair). Reads diversity from the
run's metrics json; recomputes eps-MSE from the saved FNO checkpoints.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusion import GaussianDiffusion
from fno_ddpm import FNODenoiser
from train_resinv_overnight import band_limited_noise, DEVICE

RES = [28, 40, 56]
T = 1000
TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
diffusion = GaussianDiffusion(T, device=DEVICE)


def noise_fn(mode, shape):
    return torch.randn(shape, device=DEVICE) if mode == "white" else band_limited_noise(shape, DEVICE)


def load(mode):
    m = FNODenoiser(width=48, n_blocks=4, im_shape=(28, 28), ksize=15).to(DEVICE)
    m.load_state_dict(torch.load(f"checkpoints/ddpm_fno_{mode}.pth", map_location=DEVICE))
    m.eval()
    return m


@torch.no_grad()
def eps_mse(model, mode, res, nb=15, bs=64):
    dl = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=TF), batch_size=bs)
    tot, n = 0.0, 0
    for i, (x, _) in enumerate(dl):
        if i >= nb:
            break
        x = x.to(DEVICE)
        if res != 28:
            x = F.interpolate(x, size=res, mode="bilinear", align_corners=False)
        t = torch.randint(0, T, (x.size(0),), device=DEVICE)
        loss = diffusion.p_losses(model, x, t, noise=noise_fn(mode, x.shape))
        tot += loss.item() * x.size(0); n += x.size(0)
    return tot / n


mse = {}
for mode in ["white", "colored"]:
    m = load(mode)
    mse[mode] = {r: eps_mse(m, mode, r) for r in RES}
    print(mode, "eps-MSE:", {r: round(mse[mode][r], 4) for r in RES})

J = json.load(open("figures/resinv_metrics.json"))
div = {mode: {int(r): J[mode]["metrics"][r]["diversity"] for r in J[mode]["metrics"]} for mode in ["white", "colored"]}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for mode, c in [("white", "tab:blue"), ("colored", "tab:orange")]:
    a1.plot(RES, [mse[mode][r] for r in RES], marker="o", color=c, label=f"{mode} noise")
    rs = sorted(div[mode])
    a2.plot(rs, [div[mode][r] for r in rs], marker="o", color=c, label=f"{mode} noise")
a1.set_title("Noise-prediction MSE vs resolution"); a1.set_xlabel("resolution"); a1.set_ylabel("eps-MSE (own noise)")
a2.set_title("Sample diversity vs resolution"); a2.set_xlabel("resolution"); a2.set_ylabel("digit diversity (higher=better)")
for a in (a1, a2):
    a.grid(alpha=0.3); a.legend(); a.margins(y=0.15)          # autoscaled/zoomed y-axis
fig.suptitle("White vs. band-limited (colored) noise — trends across resolution", y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig("figures/resinv_mse_diversity_zoom.png", dpi=150, bbox_inches="tight")
print("saved figures/resinv_mse_diversity_zoom.png")
