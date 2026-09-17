"""Faces noise-term ablation on the ORIGINAL-DDPM face model (CelebA-HQ 256).

Loads pretrained `google/ddpm-celebahq-256` — the diffusers port of the original
DDPM model class + dataset from Ho et al. 2020 (CelebA-HQ 256x256) — and runs the
reverse process TWICE from the SAME initial noise:
  (a) standard DDPM       -> keep the sigma_t * z term
  (b) noise term removed  -> deterministic reverse step (mean only)
then saves a side-by-side comparison. SAMPLING ONLY, no training.

The reverse step is written out explicitly (same math as our MNIST/spiral code) so
the `add_noise` toggle is unambiguous and version-robust across diffusers releases.

Run (env already built with torch+diffusers):
    uv run python faces_noise_ablation.py
Local Mac fallback: set N=2 to fit MPS memory.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

MODEL_ID = "google/ddpm-celebahq-256"
N = int(os.environ.get("N_FACES", 6))   # faces per row; use 2 on a 16GB M1 (memory)
SEED = 0       # same initial noise for the with/without comparison
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"loading {MODEL_ID} on {DEVICE} ...")
unet = UNet2DModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
sched = DDPMScheduler.from_pretrained(MODEL_ID)

T = sched.config.num_train_timesteps          # 1000
abar = sched.alphas_cumprod.to(DEVICE)        # [T]
betas = sched.betas.to(DEVICE)
alphas = 1.0 - betas
RES = unet.config.sample_size                 # 256
CH = unet.config.in_channels                  # 3


@torch.no_grad()
def sample(add_noise):
    """Reference DDPM reverse process: predict x0 -> CLIP to [-1,1] -> posterior
    mean -> (+ sigma_t z if add_noise). The x0 clipping is clip_sample=True from
    Ho et al.'s implementation; it keeps the deterministic path bounded."""
    g = torch.Generator(device=DEVICE).manual_seed(SEED)   # identical x_T both runs
    x = torch.randn(N, CH, RES, RES, generator=g, device=DEVICE)
    desc = "with-noise" if add_noise else "no-noise"
    for t in tqdm(range(T - 1, -1, -1), desc=desc, leave=False):
        eps = unet(x, torch.tensor(t, device=DEVICE)).sample
        abar_t, beta_t, alpha_t = abar[t], betas[t], alphas[t]

        x0 = (x - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)
        x0 = x0.clamp(-1, 1)                            # clip_sample=True (the fix)

        abar_prev = abar[t - 1] if t > 0 else torch.ones_like(abar_t)
        coef_x0 = beta_t * torch.sqrt(abar_prev) / (1 - abar_t)
        coef_xt = (1 - abar_prev) * torch.sqrt(alpha_t) / (1 - abar_t)
        mean = coef_x0 * x0 + coef_xt * x              # posterior mean q(x_{t-1}|x_t,x0)

        if t > 0 and add_noise:
            var = beta_t * (1 - abar_prev) / (1 - abar_t)
            noise = torch.randn(x.shape, generator=g, device=DEVICE)
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean                                   # t==0, or noise term removed
    return x


def to_img(x):
    return ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()


print("sampling WITH noise term (standard DDPM) ...")
on = to_img(sample(add_noise=True))
print("sampling WITHOUT noise term (deterministic) ...")
off = to_img(sample(add_noise=False))

fig, axes = plt.subplots(2, N, figsize=(N * 2.0, 4.4))
for i in range(N):
    axes[0, i].imshow(on[i]);  axes[0, i].axis("off")
    axes[1, i].imshow(off[i]); axes[1, i].axis("off")
for row, lab in [(0, "with noise term\n(standard DDPM)"), (1, "noise term\nremoved")]:
    axes[row, 0].axis("on")
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
    axes[row, 0].set_ylabel(lab, fontsize=10)
fig.suptitle("CelebA-HQ 256 DDPM: with vs without the noise term", y=1.0)
fig.tight_layout()
path = f"{OUT}/faces_ddpm_noise_ablation.png"
fig.savefig(path, dpi=140, bbox_inches="tight")
print("saved", path)
