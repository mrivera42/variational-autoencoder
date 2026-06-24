import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

from model import VAE

# set hyperparameters
BATCH_SIZE=128 
LR=1e-3 
NUM_EPOCHS = 10 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# loss function
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

        for x,_ in test_loader:
            x = x.to(DEVICE).view(x.size(0),-1)
            recon, mu, logvar = model(x)
            test_loss += loss_function(recon, x, mu, logvar).item()
        test_loss /= len(test_loader.dataset)
    return test_loss



    
if __name__ == "__main__":

    config = {
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "num_epochs": NUM_EPOCHS,
        "in_dim": 28 * 28,
        "trunk_dim": 400,
        "latent_dim": 20,
        "device": DEVICE,
        "optimizer": "Adam",
        "recon_loss": "BCE(sum)",
    }

    wandb.init(project="variational-autoencoder", config=config)

    # load train and test sets
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # load model
    model = VAE(
        in_dim=config["in_dim"],
        trunk_dim=config["trunk_dim"],
        latent_dim=config["latent_dim"],
    ).to(DEVICE)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):

        train_loss = train_loop(model, train_loader, optimizer)
        test_loss = test_loop(model, test_loader)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
        })

        print(f"epoch {epoch}, train loss {train_loss:.2f}, test_loss: {test_loss:.2f}")

    wandb.finish()
