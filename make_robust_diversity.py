"""Robustness check for the diversity claim (white collapses, colored holds).
The headline metric is entropy of the MEAN softmax, which can confound true
digit-variety with classifier-uncertainty. Here we recompute with a HARD
argmax-class histogram (classifier's committed label per sample) and report
#distinct classes + recognizability, so the claim rests on >1 measure.

    uv run python make_robust_diversity.py
"""
import json
import numpy as np
import torch
import torch.nn.functional as F

import train_resinv_overnight as R
from diffusion import GaussianDiffusion
from fno_ddpm import FNODenoiser

RES = [28, 40, 56]
N = 16
diffusion = GaussianDiffusion(1000, device=R.DEVICE)
clf = R.train_classifier()


def load(mode):
    m = FNODenoiser(width=48, n_blocks=4, im_shape=(28, 28), ksize=15).to(R.DEVICE)
    m.load_state_dict(torch.load(f"checkpoints/ddpm_fno_{mode}.pth", map_location=R.DEVICE))
    m.eval(); return m


@torch.no_grad()
def measure(imgs):
    x = torch.tensor(imgs, device=R.DEVICE).unsqueeze(1) * 2 - 1
    if x.shape[-1] != 28:
        x = F.interpolate(x, size=28, mode="bilinear", align_corners=False)
    p = F.softmax(clf(x), dim=1)
    recog = p.max(1).values.mean().item()
    soft = (-(p.mean(0) * (p.mean(0) + 1e-9).log()).sum() / np.log(10)).item()   # original
    hard_lbls = p.argmax(1)
    counts = torch.bincount(hard_lbls, minlength=10).float()
    ph = counts / counts.sum()
    hard = (-(ph * (ph + 1e-9).log()).sum() / np.log(10)).item()                  # argmax-hist entropy
    ndist = int((counts > 0).sum().item())                                        # #distinct classes
    return dict(recog=round(recog, 3), soft_div=round(soft, 3),
                hard_div=round(hard, 3), distinct=ndist)


out = {}
for mode in ["white", "colored"]:
    m = load(mode)
    out[mode] = {}
    for res in RES:
        imgs = R.sample(m, diffusion, res, mode, n=N)
        out[mode][res] = measure(imgs)
        print(f"{mode}@{res} (n={N}):", out[mode][res], flush=True)

print("\n=== ROBUSTNESS SUMMARY (n=%d) ===" % N)
for mode in ["white", "colored"]:
    print(mode, {r: out[mode][r] for r in RES})
json.dump(out, open("figures/robust_diversity.json", "w"), indent=2)
print("saved figures/robust_diversity.json")
print("ALL DONE", flush=True)
