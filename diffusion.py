"""Gaussian diffusion process (DDPM), space-agnostic.

This holds the *math* of DDPM and is deliberately independent of what the
denoiser looks like or what space we diffuse in. The same `GaussianDiffusion`
object drives both:
  - pixel-space DDPM (data = images, denoiser = UNet), and
  - latent DDPM      (data = VAE latents, denoiser = LatentDenoiser).

A denoiser `model(x_t, t)` takes a noised sample `x_t` and an integer timestep
`t` (shape [B]) and predicts the noise `eps` that was added (same shape as x_t).

Notation follows Ho et al. 2020:
  beta_t                      : variance added at step t
  alpha_t      = 1 - beta_t
  alpha_bar_t  = prod_{s<=t} alpha_s
  q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)

NOTE (self-scaffold): the four methods tagged `# SCAFFOLD` below are the DDPM
learning exercises. To redo this yourself, delete their bodies (keep the
signatures + docstrings) and reimplement. See SCAFFOLD_NOTES.md.
"""

import torch
import torch.nn.functional as F


def _extract(coeffs, t, x_shape):
    """Gather per-timestep scalars into a broadcastable batch tensor.

    coeffs : 1-D tensor of length num_timesteps (e.g. self.sqrt_alpha_bars)
    t      : LongTensor [B] of timestep indices, one per batch element
    x_shape: the shape of the tensor we want to broadcast against, e.g. x_t.shape

    Returns coeffs[t] reshaped to [B, 1, 1, ...] so it multiplies x elementwise.
    Provided for you — this indexing is fiddly and not the point of the exercise.
    """
    b = t.shape[0]
    out = coeffs.gather(0, t)                            # [B]
    return out.reshape(b, *([1] * (len(x_shape) - 1)))  # [B, 1, 1, ...]


class GaussianDiffusion:

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

        # --- noise schedule -------------------------------------------------
        betas = self.make_beta_schedule(num_timesteps, beta_start, beta_end).to(device)

        # --- derived quantities (precomputed once, stored as [T] tensors) ----
        # SCAFFOLD: computing these buffers is part of the exercise.
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)                       # prod of alphas up to t
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)     # alpha_bar_{t-1}, with abar_{-1}=1

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        # coefficients used in q_sample: x_t = sqrt(abar) x0 + sqrt(1-abar) eps
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

        # coefficients used in the reverse (p_sample) posterior mean
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # true posterior variance of q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    @staticmethod
    def make_beta_schedule(num_timesteps, beta_start, beta_end):
        """Return a 1-D tensor of betas, shape [num_timesteps].

        SCAFFOLD: linear schedule from the DDPM paper — evenly spaced from
        beta_start to beta_end.
        """
        return torch.linspace(beta_start, beta_end, num_timesteps)

    # ---- forward process q(x_t | x_0) --------------------------------------
    def q_sample(self, x_0, t, noise=None):
        """Diffuse x_0 to x_t in one shot (the closed-form forward process).

        x_0   : clean sample, shape [B, ...]
        t     : LongTensor [B]
        noise : optional eps ~ N(0, I) of shape like x_0 (sample one if None)

        Returns x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        # SCAFFOLD
        sqrt_ab = _extract(self.sqrt_alpha_bars, t, x_0.shape)
        sqrt_1mab = _extract(self.sqrt_one_minus_alpha_bars, t, x_0.shape)
        return sqrt_ab * x_0 + sqrt_1mab * noise

    # ---- training objective -------------------------------------------------
    def p_losses(self, model, x_0, t, noise=None):
        """The simple DDPM loss for a batch: MSE between true and predicted noise."""
        if noise is None:
            noise = torch.randn_like(x_0)
        # SCAFFOLD
        x_t = self.q_sample(x_0, t, noise)
        predicted = model(x_t, t)               # denoiser predicts the noise eps
        return F.mse_loss(predicted, noise)

    # ---- reverse process p(x_{t-1} | x_t) ----------------------------------
    @torch.no_grad()
    def p_sample(self, model, x_t, t, add_noise=True):
        """One reverse denoising step: given x_t, sample x_{t-1}.

        t : LongTensor [B] (same value across the batch during sampling).

        mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_pred)
        Add noise sqrt(posterior_variance_t) * z (z ~ N(0, I)), except at t == 0.

        add_noise : if False, drop the `sigma_t * z` term so the reverse step is
        deterministic (the mean only). This is the "comment out the noise
        addition" experiment — set False to see what generation produces without it.
        """
        # SCAFFOLD
        eps_pred = model(x_t, t)
        betas_t = _extract(self.betas, t, x_t.shape)
        sqrt_1mab_t = _extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t, x_t.shape)

        mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_1mab_t * eps_pred)

        if (t == 0).all():
            return mean
        if not add_noise:                 # experiment: noise-injection term removed
            return mean
        var_t = _extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var_t) * noise

    @torch.no_grad()
    def sample(self, model, shape, progress=False, add_noise=True):
        """Full ancestral sampling loop: x_T ~ N(0, I) -> ... -> x_0.

        shape : desired output shape, e.g. (n, 1, 28, 28) or (n, latent_dim).
        progress : show a tqdm bar over the reverse steps (sampling is slow).
        add_noise : passed to p_sample; set False to remove the noise-injection
        term from every reverse step (the "comment out the noise" experiment).
        Returns the final x_0.
        """
        # SCAFFOLD
        x = torch.randn(shape, device=self.device)
        steps = reversed(range(self.num_timesteps))
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps, total=self.num_timesteps, desc="sampling", leave=False)
        for step in steps:
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, add_noise=add_noise)
        return x
