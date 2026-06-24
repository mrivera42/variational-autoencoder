import torch 
import torch.nn as nn 


class VAE(nn.Module):

    def __init__(self, in_dim, trunk_dim, latent_dim):

        super().__init__()
        self.in_shape = in_dim
        self.trunk_dim = trunk_dim
        self.latent_dim = latent_dim

        # define my layers 
        self.fc_encoder_trunk = nn.Linear(in_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.fc_decoder_hidden = nn.Linear(latent_dim, 400)
        self.fc_decoder_out = nn.Linear(400, in_dim)
        self.relu = nn.ReLU()

    def encode(self, x):

        # shared trunk, two heads 
        x = self.relu(self.fc_encoder_trunk(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


    def reparameterize(self, mu, logvar):

        # rewrite sample as a determinstic function of mu and logvar 
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn(logvar.shape,device=sigma.device)
        z = mu + sigma * eps

        return z 

    def decode(self, z):

        # take z, reconstruct original dimensions 
        h = self.relu(self.fc_decoder_hidden(z))
        out = torch.sigmoid(self.fc_decoder_out(h))

        return out

    def forward(self, x):

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return recon, mu, logvar






if __name__ == "__main__":

    model = VAE(
        in_dim=600,
        trunk_dim=400,
        latent_dim=4
    )
    x = torch.randn(4, 600)
    print('x shape: ', x.shape)
    recon, mu, logvar = model(x)
    print(f'mu shape: {mu.shape}')
    print(f'logvar shape: {logvar.shape}')
    print('recon shape: ', recon.shape)
