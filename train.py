import json
import os

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

from model import VAE


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"


def loss_function(recon_x, x, mu, logvar):
    """returns BCE + KL"""
    recon_term = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_term = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_term + kl_term

# train loop 
def train_loop(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):

        x = x.to(DEVICE)
        x = x.view(x.size(0),-1)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = loss_function(recon, x, mu, logvar)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_loader.dataset)


def test_loop(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE).view(x.size(0), -1)
            recon, mu, logvar = model(x)
            test_loss += loss_function(recon, x, mu, logvar).item()
    return test_loss / len(test_loader.dataset)


def run_experiment(config):
    """Train a VAE according to config, log to W&B, save checkpoint.

    config keys: name, batch_size, lr, num_epochs, in_dim, trunk_dim,
                 latent_dim, n_hidden_layers.

    Returns a dict with per-epoch losses and final losses.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    wandb.init(
        project="variational-autoencoder",
        name=config["name"],
        config={**config, "device": DEVICE, "optimizer": "Adam", "recon_loss": "BCE(sum)"},
        reinit=True,
    )

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = VAE(
        in_dim=config["in_dim"],
        trunk_dim=config["trunk_dim"],
        latent_dim=config["latent_dim"],
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_losses = []
    test_losses = []
    for epoch in range(config["num_epochs"]):
        train_loss = train_loop(model, train_loader, optimizer)
        test_loss = test_loop(model, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
        print(f"[{config['name']}] epoch {epoch}, train {train_loss:.2f}, test {test_loss:.2f}")

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


BASELINE_CONFIG = {
    "name": "baseline_z20",
    "batch_size": 128,
    "lr": 1e-3,
    "num_epochs": 10,
    "in_dim": 28 * 28,
    "trunk_dim": 400,
    "latent_dim": 20,
}


if __name__ == "__main__":
    run_experiment(BASELINE_CONFIG)