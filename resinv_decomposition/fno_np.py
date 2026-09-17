"""Numpy port of Max's FNODenoiser + GaussianDiffusion (no torch required).

Mirrors:
  fno_ddpm.py       FNODenoiser / FNOBlock
  spectral_conv.py  SpectralConv2d.forward + resize_spectral
  ddpm_model.py     SinusoidalTimeEmbedding
  diffusion.py      GaussianDiffusion (linear beta schedule, eps-pred)
"""
import numpy as np, math
from loadpth import load

# ---------------------------------------------------------------- primitives
def silu(x):  return x / (1.0 + np.exp(-x))

def group_norm(x, w, b, groups=8, eps=1e-5):
    B, C, H, W = x.shape
    xg = x.reshape(B, groups, C // groups, H, W)
    m = xg.mean(axis=(2, 3, 4), keepdims=True)
    v = xg.var(axis=(2, 3, 4), keepdims=True)
    xg = (xg - m) / np.sqrt(v + eps)
    x = xg.reshape(B, C, H, W)
    return x * w[None, :, None, None] + b[None, :, None, None]

def conv1x1(x, w, b):
    # w: [out,in,1,1]
    return np.einsum('bchw,oc->bohw', x, w[:, :, 0, 0]) + b[None, :, None, None]

def sinusoidal_time_emb(t, dim=128):
    half = dim // 2
    freqs = np.exp(-math.log(10000) * np.arange(half, dtype=np.float64) / (half - 1))
    args = t[:, None].astype(np.float64) * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=-1).astype(np.float32)

def resize_spectral(w, target_h, target_w):
    """Verbatim port of spectral_conv.resize_spectral."""
    H_w, W_w = w.shape[-2], w.shape[-1]
    if target_w > W_w:
        w = np.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, target_w - W_w)])
    elif target_w < W_w:
        w = w[..., :target_w]
    if target_h > H_w:
        top, bottom = w[..., :H_w // 2, :], w[..., H_w // 2:, :]
        zeros = np.zeros(w.shape[:-2] + (target_h - H_w, target_w), dtype=w.dtype)
        w = np.concatenate([top, zeros, bottom], axis=-2)
    elif target_h < H_w:
        top = w[..., :target_h // 2, :]
        bottom = w[..., -(target_h - target_h // 2):, :]
        w = np.concatenate([top, bottom], axis=-2)
    return w

def spectral_conv(x, w_spec, bias):
    H_in = x.shape[-2]
    X = np.fft.rfft2(x, norm='backward')                 # [B,C,H,Wf]
    w = resize_spectral(w_spec, X.shape[-2], X.shape[-1])  # [O,C,H,Wf]
    out = np.einsum('bchw,ochw->bohw', X, w)
    out = np.fft.irfft2(out, s=(H_in, H_in), norm='backward')
    return out + bias[None, :, None, None]

# ---------------------------------------------------------------- the model
class FNODenoiserNP:
    def __init__(self, sd, n_blocks=4, time_dim=128):
        self.sd, self.n_blocks, self.time_dim = sd, n_blocks, time_dim

    def __call__(self, x, t):
        sd = self.sd
        te = sinusoidal_time_emb(t, self.time_dim)
        te = te @ sd['time_mlp.1.weight'].T + sd['time_mlp.1.bias']
        te = silu(te)
        te = te @ sd['time_mlp.3.weight'].T + sd['time_mlp.3.bias']
        h = conv1x1(x, sd['lift.weight'], sd['lift.bias'])
        for i in range(self.n_blocks):
            p = f'blocks.{i}.'
            hs = spectral_conv(h, sd[p + 'spectral.w_spectral'], sd[p + 'spectral.bias'])
            hp = conv1x1(h, sd[p + 'pointwise.weight'], sd[p + 'pointwise.bias'])
            hh = hs + hp
            tp = te @ sd[p + 'time_proj.weight'].T + sd[p + 'time_proj.bias']
            hh = hh + tp[:, :, None, None]
            hh = silu(group_norm(hh, sd[p + 'norm.weight'], sd[p + 'norm.bias']))
            h = h + hh
        return conv1x1(h, sd['proj.weight'], sd['proj.bias'])

# ---------------------------------------------------------------- diffusion
class Diffusion:
    def __init__(self, T=1000, b0=1e-4, b1=0.02):
        self.T = T
        betas = np.linspace(b0, b1, T)
        alphas = 1.0 - betas
        ab = np.cumprod(alphas)
        ab_prev = np.concatenate([[1.0], ab[:-1]])
        self.betas, self.alphas, self.ab = betas, alphas, ab
        self.sqrt_ab = np.sqrt(ab)
        self.sqrt_1mab = np.sqrt(1.0 - ab)
        self.sqrt_recip_alphas = np.sqrt(1.0 / alphas)
        self.post_var = betas * (1.0 - ab_prev) / (1.0 - ab)

def load_model(path):
    return FNODenoiserNP(load(path))
