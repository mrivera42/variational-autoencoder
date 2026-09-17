"""Lambda sweep: scale the reverse-process noise term by lambda in [0, 1].

    x_{t-1} = posterior_mean + lambda * sigma_t * z

lambda = 1.0 -> standard DDPM (full noise);  lambda = 0.0 -> deterministic (no noise).
Intermediate lambdas show the transition from sharp/diverse -> degraded -> collapse.

Efficiency trick: the different lambdas are the BATCH dimension, and they SHARE the
same initial noise x_T and the same per-step noise z (only the lambda multiplier
differs). So one 1000-step pass produces the whole sweep, and any differences are
purely due to lambda.

NOTE: lambda is a NON-STANDARD experimental knob (akin to DDIM's eta), not part of
textbook DDPM. Uses the reference clipped sampler (clip_sample=True).

    uv run python faces_lambda_sweep.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

MODEL_ID = "google/ddpm-celebahq-256"
LAMBDAS = [float(v) for v in os.environ.get("LAMBDAS", "1.0,0.75,0.5,0.25,0.0").split(",")]
SEED = 0
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"loading {MODEL_ID} on {DEVICE} ; lambdas={LAMBDAS}")
unet = UNet2DModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
sched = DDPMScheduler.from_pretrained(MODEL_ID)
T = sched.config.num_train_timesteps
abar = sched.alphas_cumprod.to(DEVICE)
betas = sched.betas.to(DEVICE)
alphas = 1.0 - betas
RES = unet.config.sample_size
CH = unet.config.in_channels

lam = torch.tensor(LAMBDAS, device=DEVICE).view(-1, 1, 1, 1)   # [B,1,1,1]
B = lam.shape[0]


@torch.no_grad()
def sweep():
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    x_init = torch.randn(1, CH, RES, RES, generator=g, device=DEVICE)   # shared x_T
    x = x_init.repeat(B, 1, 1, 1)
    for t in tqdm(range(T - 1, -1, -1), desc="lambda-sweep", leave=False):
        eps = unet(x, torch.tensor(t, device=DEVICE)).sample
        abar_t, beta_t, alpha_t = abar[t], betas[t], alphas[t]
        x0 = ((x - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)).clamp(-1, 1)
        abar_prev = abar[t - 1] if t > 0 else torch.ones_like(abar_t)
        coef_x0 = beta_t * torch.sqrt(abar_prev) / (1 - abar_t)
        coef_xt = (1 - abar_prev) * torch.sqrt(alpha_t) / (1 - abar_t)
        mean = coef_x0 * x0 + coef_xt * x
        if t > 0:
            z = torch.randn(1, CH, RES, RES, generator=g, device=DEVICE)   # shared z
            var = beta_t * (1 - abar_prev) / (1 - abar_t)
            x = mean + lam * torch.sqrt(var) * z                           # scaled per lambda
        else:
            x = mean
    return x


imgs = ((sweep().clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()

fig, axes = plt.subplots(1, B, figsize=(B * 2.1, 2.6))
for i in range(B):
    axes[i].imshow(imgs[i]); axes[i].axis("off")
    axes[i].set_title(f"$\\lambda$ = {LAMBDAS[i]:g}", fontsize=11)
fig.suptitle("CelebA-HQ 256 DDPM: reverse-process noise scaled by $\\lambda$", fontsize=12)
fig.tight_layout()
path = f"{OUT}/faces_lambda_sweep.png"
fig.savefig(path, dpi=140, bbox_inches="tight")
print("saved", path)
