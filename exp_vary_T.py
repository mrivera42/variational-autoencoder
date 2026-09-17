"""Experiment 2 (advisor bullet 1): vary T, retrain, and sample.

Retrain the latent DDPM with different total diffusion steps T (same linear beta
schedule 1e-4 -> 0.02), then generate from each. We use the LATENT model because
it trains in ~5 min, so a full T-sweep is cheap; the pixel model takes ~3.25 h
per run and is better left for an overnight sweep.

Honest caveat: with the beta endpoints held fixed, a smaller T means the forward
process adds less total noise, so x_T is not fully N(0, I). Part of what we observe.

    uv run python exp_vary_T.py            # logs each run to W&B (project ddpm-mnist)
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train_ddpm import run_experiment, load_vae
from inference_ddpm import load_ddpm

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
N = 8               # samples per row
SEED = 0
EPOCHS = 40         # latent converges well before this; keeps the sweep quick
T_VALUES = [10, 50, 100, 250, 1000]


def config_for(T):
    return {
        "name": f"ddpm_latent_T{T}", "space": "latent", "vae_name": "baseline_z20",
        "batch_size": 128, "lr": 1e-3, "num_epochs": EPOCHS,
        "num_timesteps": T, "beta_start": 1e-4, "beta_end": 0.02,
        "time_dim": 128, "hidden_dim": 256,
    }


@torch.no_grad()
def sample_decode(run_name):
    model, diffusion, cfg = load_ddpm(run_name)
    vae, _ = load_vae(cfg["vae_name"])
    torch.manual_seed(SEED)
    z0 = diffusion.sample(model, (N, cfg["latent_dim"]))
    return vae.decode(z0).view(N, 28, 28).cpu().numpy()


def main():
    results = []      # (T, final_test_loss, train_seconds)
    for T in T_VALUES:
        print(f"\n=== training ddpm_latent_T{T} (T={T}) — logging to W&B project ddpm-mnist ===")
        t0 = time.time()
        res = run_experiment(config_for(T))       # run_experiment logs to W&B
        train_secs = time.time() - t0
        print(f"    [T={T}] training time: {train_secs:.1f}s ({train_secs/60:.1f} min), "
              f"final test loss {res['final_test_loss']:.4f}")
        results.append((T, res["final_test_loss"], train_secs))

    fig, axes = plt.subplots(len(T_VALUES), N, figsize=(N * 1.3, len(T_VALUES) * 1.5))
    for r, T in enumerate(T_VALUES):
        imgs = sample_decode(f"ddpm_latent_T{T}")
        for i in range(N):
            axes[r, i].imshow(imgs[i], cmap="gray")
            axes[r, i].axis("off")
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"T = {T}", fontsize=10)
    fig.suptitle("Effect of number of diffusion steps T — Latent DDPM (retrained per T)", y=1.005)
    fig.tight_layout()
    fig.savefig(f"{FIG}/exp2_vary_T_latent.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/exp2_vary_T_latent.png")

    total = sum(s for _, _, s in results)
    print("\n  T   | final test loss | training time")
    print("  --- | --------------- | -------------")
    for T, loss, secs in results:
        print(f"  {T:>4} |     {loss:.4f}      | {secs/60:.1f} min")
    print(f"  total training time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
