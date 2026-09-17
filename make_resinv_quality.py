"""Clean quality metric: sample diversity + high-frequency energy vs resolution,
white vs band-limited (colored) noise. HF energy is high for static (white @56),
low for smooth/coherent (colored @56). Diversity read from the run's json.
"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import train_resinv_overnight as R
from diffusion import GaussianDiffusion
from fno_ddpm import FNODenoiser

RES = [28, 40, 56]
diffusion = GaussianDiffusion(1000, device=R.DEVICE)


def load(mode):
    m = FNODenoiser(width=48, n_blocks=4, im_shape=(28, 28), ksize=15).to(R.DEVICE)
    m.load_state_dict(torch.load(f"checkpoints/ddpm_fno_{mode}.pth", map_location=R.DEVICE))
    m.eval(); return m


def hf_fraction(imgs, cutoff=0.5):
    x = torch.tensor(imgs, dtype=torch.float32)          # [n, H, W] in [0,1]
    H, W = x.shape[-2], x.shape[-1]
    X = torch.fft.rfft2(x); p = X.abs() ** 2
    fh = torch.fft.fftfreq(H).abs() * 2                   # normalized to Nyquist=1
    fw = torch.fft.rfftfreq(W).abs() * 2
    r = torch.sqrt(fh[:, None] ** 2 + fw[None, :] ** 2)
    hf = (p * (r > cutoff)).sum(dim=(1, 2)) / (p.sum(dim=(1, 2)) + 1e-8)
    return hf.mean().item()


hf = {}
for mode in ["white", "colored"]:
    m = load(mode)
    hf[mode] = {res: hf_fraction(R.sample(m, diffusion, res, mode, n=12)) for res in RES}
    print(mode, "HF-energy:", {r: round(hf[mode][r], 3) for r in RES}, flush=True)

J = json.load(open("figures/resinv_metrics.json"))
div = {mode: {int(r): J[mode]["metrics"][r]["diversity"] for r in J[mode]["metrics"]} for mode in ["white", "colored"]}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for mode, c in [("white", "tab:blue"), ("colored", "tab:orange")]:
    a1.plot(RES, [hf[mode][r] for r in RES], marker="o", color=c, label=f"{mode} noise")
    rs = sorted(div[mode]); a2.plot(rs, [div[mode][r] for r in rs], marker="o", color=c, label=f"{mode} noise")
a1.set_title("High-frequency energy of samples\n(high = static, low = smooth)")
a1.set_xlabel("resolution"); a1.set_ylabel("fraction of energy above ½ Nyquist")
a2.set_title("Sample diversity\n(higher = less collapse)")
a2.set_xlabel("resolution"); a2.set_ylabel("digit diversity")
for a in (a1, a2):
    a.grid(alpha=0.3); a.legend(); a.margins(y=0.15)
fig.suptitle("White vs. band-limited (colored) noise — sample quality across resolution", y=1.03, fontsize=13)
fig.tight_layout()
fig.savefig("figures/resinv_quality_zoom.png", dpi=150, bbox_inches="tight")
print("saved figures/resinv_quality_zoom.png", flush=True)
print("ALL DONE", flush=True)
