"""Train a DDPM on MNIST — pixel-space or latent-space, selected by config.

Pixel space  (config["space"] == "pixel"):
    diffuse raw images in [-1, 1], denoiser = UNet.

Latent space (config["space"] == "latent"):
    encode images with a *frozen* pretrained VAE, diffuse the latents,
    denoiser = LatentDenoiser. Reuses the VAE you trained in this repo
    (config["vae_name"], e.g. "baseline_z20").

The training/eval loops and all wiring are done. You implement the diffusion
math in diffusion.py and the model forwards in ddpm_model.py; everything here
just calls into those.
"""

import json
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

from diffusion import GaussianDiffusion
from ddpm_model import UNet, LatentDenoiser
from model import VAE


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"


# ---------------------------------------------------------------------------
# VAE reuse (latent DDPM only) — provided, no need to edit
# ---------------------------------------------------------------------------
def load_vae(vae_name):
    """Load a trained VAE checkpoint, put it in eval mode, freeze its params."""
    cfg_path = os.path.join(CHECKPOINT_DIR, f"{vae_name}.json")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{vae_name}.pth")
    with open(cfg_path) as f:
        vae_cfg = json.load(f)
    vae = VAE(
        in_dim=vae_cfg["in_dim"],
        trunk_dim=vae_cfg["trunk_dim"],
        latent_dim=vae_cfg["latent_dim"],
    ).to(DEVICE)
    vae.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, vae_cfg


@torch.no_grad()
def encode_to_latent(vae, x):
    """Images -> VAE latents used as the diffusion target.

    x is a flat image batch [B, 784] in [0, 1] (VAE was trained on ToTensor).
    We use the posterior mean mu as the latent (deterministic). If you'd rather
    diffuse reparameterized samples, swap in vae.reparameterize(mu, logvar).
    """
    mu, logvar = vae.encode(x)
    return mu


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def make_loaders(config):
    """Return (train_loader, test_loader).

    Pixel DDPM diffuses images, which we normalize to [-1, 1] to match the
    N(0, I) prior the reverse process starts from. Latent DDPM diffuses VAE
    latents, so images stay in [0, 1] (that's what the VAE expects) and the
    normalization happens implicitly via the encoder.
    """
    if config["space"] == "pixel":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),   # [0,1] -> [-1,1]
        ])
    else:
        transform = transforms.ToTensor()           # [0,1], VAE's training regime

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
    return train_loader, test_loader


def prepare_batch(x, config, vae):
    """Turn a raw image batch into the tensor we actually diffuse.

    pixel : keep images as [B, 1, 28, 28] (UNet is convolutional).
    latent: flatten to [B, 784] and encode with the frozen VAE -> [B, latent_dim].
    """
    x = x.to(DEVICE)
    if config["space"] == "pixel":
        return x                                     # [B, 1, 28, 28]
    return encode_to_latent(vae, x.view(x.size(0), -1))  # [B, latent_dim]


# ---------------------------------------------------------------------------
# train / eval loops — call into diffusion.p_losses (which you implement)
# ---------------------------------------------------------------------------
def sample_timesteps(batch_size, num_timesteps):
    """Uniformly sample one timestep per batch element: LongTensor [B]."""
    return torch.randint(0, num_timesteps, (batch_size,), device=DEVICE)


def train_loop(model, diffusion, train_loader, optimizer, config, vae, epoch=None):
    model.train()
    running = 0.0
    desc = f"train e{epoch}" if epoch is not None else "train"
    pbar = tqdm(train_loader, desc=desc, leave=False)
    for x, _ in pbar:
        x0 = prepare_batch(x, config, vae)
        t = sample_timesteps(x0.size(0), diffusion.num_timesteps)

        optimizer.zero_grad()
        loss = diffusion.p_losses(model, x0, t)
        loss.backward()
        optimizer.step()
        running += loss.item() * x0.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / len(train_loader.dataset)


@torch.no_grad()
def test_loop(model, diffusion, test_loader, config, vae):
    model.eval()
    running = 0.0
    for x, _ in tqdm(test_loader, desc="test", leave=False):
        x0 = prepare_batch(x, config, vae)
        t = sample_timesteps(x0.size(0), diffusion.num_timesteps)
        loss = diffusion.p_losses(model, x0, t)
        running += loss.item() * x0.size(0)
    return running / len(test_loader.dataset)


# ---------------------------------------------------------------------------
# build the denoiser for a given config
# ---------------------------------------------------------------------------
def build_model(config):
    if config["space"] == "pixel":
        return UNet(
            in_ch=1,
            base_ch=config["base_ch"],
            time_dim=config["time_dim"],
        ).to(DEVICE)
    return LatentDenoiser(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        time_dim=config["time_dim"],
    ).to(DEVICE)


def run_experiment(config):
    """Train one DDPM according to config, log to W&B, save checkpoint.

    config keys (common): name, space ("pixel"|"latent"), batch_size, lr,
        num_epochs, num_timesteps, beta_start, beta_end, time_dim.
    pixel  adds: base_ch.
    latent adds: vae_name, latent_dim, hidden_dim.

    Returns a dict of per-epoch and final losses (parallels train.run_experiment).
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    wandb.init(
        project="ddpm-mnist",
        name=config["name"],
        config={**config, "device": DEVICE, "optimizer": "Adam", "loss": "MSE(eps)"},
        reinit=True,
    )

    vae = None
    if config["space"] == "latent":
        vae, vae_cfg = load_vae(config["vae_name"])
        # keep the denoiser's latent_dim in sync with the VAE we loaded
        config["latent_dim"] = vae_cfg["latent_dim"]

    train_loader, test_loader = make_loaders(config)

    diffusion = GaussianDiffusion(
        num_timesteps=config["num_timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        device=DEVICE,
    )
    model = build_model(config)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_losses, test_losses = [], []
    for epoch in range(config["num_epochs"]):
        train_loss = train_loop(model, diffusion, train_loader, optimizer, config, vae, epoch)
        test_loss = test_loop(model, diffusion, test_loader, config, vae)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
        print(f"[{config['name']}] epoch {epoch}, train {train_loss:.4f}, test {test_loss:.4f}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{config['name']}.pth")
    cfg_path = os.path.join(CHECKPOINT_DIR, f"{config['name']}.json")
    torch.save(model.state_dict(), ckpt_path)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    wandb.finish()

    return {
        "name": config["name"],
        "train_losses": train_losses,
        "test_losses": test_losses,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "checkpoint": ckpt_path,
    }


PIXEL_CONFIG = {
    "name": "ddpm_pixel",
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

LATENT_CONFIG = {
    "name": "ddpm_latent_z20",
    "space": "latent",
    "vae_name": "baseline_z20",   # reuse the VAE trained in this repo
    "batch_size": 128,
    "lr": 1e-3,
    "num_epochs": 30,
    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "time_dim": 128,
    "hidden_dim": 256,
}


if __name__ == "__main__":
    run_experiment(PIXEL_CONFIG)
