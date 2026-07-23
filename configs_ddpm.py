"""DDPM experiment configurations.

Each entry is a dict consumed by `train_ddpm.run_experiment(config)`. Mirrors
the structure of configs.py for the VAE. Two families:
  - pixel  : diffuse images with a UNet.
  - latent : diffuse frozen-VAE latents with an MLP denoiser.
Each varies one knob so the effect is isolated.
"""

_PIXEL_BASE = {
    "space": "pixel",
    "batch_size": 128,
    "lr": 2e-4,
    "num_epochs": 20,
    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "time_dim": 128,
    "base_ch": 64,
}

_LATENT_BASE = {
    "space": "latent",
    "vae_name": "baseline_z20",
    "batch_size": 128,
    "lr": 1e-3,
    "num_epochs": 30,
    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "time_dim": 128,
    "hidden_dim": 256,
}


def _pixel(**overrides):
    cfg = dict(_PIXEL_BASE)
    cfg.update(overrides)
    return cfg


def _latent(**overrides):
    cfg = dict(_LATENT_BASE)
    cfg.update(overrides)
    return cfg


EXPERIMENTS = [
    _pixel(name="ddpm_pixel"),                              # pixel-space baseline
    _pixel(name="ddpm_pixel_T500", num_timesteps=500),     # fewer diffusion steps
    _latent(name="ddpm_latent_z20"),                       # latent baseline (reuses baseline_z20 VAE)
    _latent(name="ddpm_latent_wide", hidden_dim=512),      # wider denoiser
    _latent(name="ddpm_latent_long", num_epochs=60),       # more epochs
]
