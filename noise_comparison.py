"""Side-by-side: white noise vs. band-limited (colored) noise, for slides.
White = grainy, pixel-scale static (all frequencies). Colored = smooth, low-pass
(fine grain removed). Same setup used in the resolution-invariance experiment.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

FIG = "figures"; os.makedirs(FIG, exist_ok=True)
RES, N, CUTOFF = 56, 5, 8
torch.manual_seed(0)


def band_limited_noise(shape, cutoff=CUTOFF):
    B, C, H, W = shape
    white = torch.randn(B, C, H, W)
    fh = torch.fft.fftfreq(H) * H
    fw = torch.fft.rfftfreq(W) * W
    r = torch.sqrt(fh[:, None] ** 2 + fw[None, :] ** 2)
    env = torch.exp(-(r / cutoff) ** 2)
    n = torch.fft.irfft2(torch.fft.rfft2(white) * env, s=(H, W))
    return n / (n.std(dim=(1, 2, 3), keepdim=True) + 1e-8)


white = torch.randn(N, 1, RES, RES)
colored = band_limited_noise((N, 1, RES, RES))

fig, axes = plt.subplots(2, N, figsize=(N * 1.6, 3.4))
for i in range(N):
    axes[0, i].imshow(white[i, 0], cmap="gray", vmin=-3, vmax=3); axes[0, i].axis("off")
    axes[1, i].imshow(colored[i, 0], cmap="gray", vmin=-3, vmax=3); axes[1, i].axis("off")
for r, lab in [(0, "white noise"), (1, "band-limited\n(colored) noise")]:
    axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    axes[r, 0].set_ylabel(lab, fontsize=12)
fig.suptitle("White noise vs. band-limited (colored) noise  —  56×56", fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig(f"{FIG}/noise_white_vs_colored.png", dpi=150, bbox_inches="tight")
print("saved figures/noise_white_vs_colored.png")
