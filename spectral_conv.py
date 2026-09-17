"""Resolution-invariant spectral convolution.

Copied verbatim from the user's graph-neural-operator repo
(`graph-neural-operator/spectral_cnn.py`) — the CNN->FNO conversion via the
convolution theorem. Kept here so the diffusion experiment is self-contained.

SpectralConv2d is a drop-in, resolution-invariant replacement for nn.Conv2d:
it multiplies in the Fourier domain and `resize_spectral` adapts the learned
weights to whatever input resolution is passed at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


def kernel_to_spectral(weight, im_shape):
    # flip to make it a true convolution instead of cross-correlation
    w_flipped = torch.flip(weight, dims=[-2, -1])

    H = im_shape[0]
    kH = weight.shape[2]
    W = im_shape[1]
    kW = weight.shape[3]
    bottom_h = (H - kH) // 2
    top_h = (H - kH) - bottom_h
    right_w = (W - kW) // 2
    left_w = (W - kW) - right_w

    w_padded = F.pad(w_flipped, pad=(left_w, right_w, top_h, bottom_h))
    w_shifted = fft.ifftshift(w_padded, dim=[-2, -1])
    w_spectral = fft.rfft2(w_shifted, norm='backward')
    return w_spectral


def resize_spectral(w, target_h, target_w):
    """Resize spectral weights to match target FFT dimensions (res-invariance)."""
    H_w, W_w = w.shape[-2], w.shape[-1]

    if target_w > W_w:
        w = F.pad(w, (0, target_w - W_w))
    elif target_w < W_w:
        w = w[..., :target_w]

    if target_h > H_w:
        top = w[..., :H_w // 2, :]
        bottom = w[..., H_w // 2:, :]
        zeros = torch.zeros(*w.shape[:-2], target_h - H_w, target_w,
                            dtype=w.dtype, device=w.device)
        w = torch.cat([top, zeros, bottom], dim=-2)
    elif target_h < H_w:
        top = w[..., :target_h // 2, :]
        bottom = w[..., -(target_h - target_h // 2):, :]
        w = torch.cat([top, bottom], dim=-2)
    return w


class SpectralConv2d(nn.Module):
    """Conv layer operating entirely in the Fourier domain; y = iFFT(FFT(x) * W)."""

    def __init__(self, conv: nn.Conv2d, im_shape: tuple):
        super().__init__()
        w = conv.weight
        self.w_spectral = nn.Parameter(kernel_to_spectral(w, im_shape))
        self.bias = conv.bias

    @classmethod
    def from_scratch(cls, in_channels, out_channels, im_shape, ksize=None):
        """Random spectral weights. ksize = # of frequency bins (locality bias)."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        if ksize is not None:
            H, W = ksize, ksize
        else:
            H, W = im_shape
        scale = 1 / (in_channels * out_channels)
        w = scale * torch.rand(out_channels, in_channels, H, W // 2 + 1, dtype=torch.cfloat)
        w[:, :, 0, 0].imag = 0.0  # DC component must be real
        obj.w_spectral = nn.Parameter(w)
        obj.bias = nn.Parameter(torch.zeros(out_channels))
        return obj

    def forward(self, x):
        H_in = x.shape[-2]
        x = fft.rfft2(x, norm='backward')
        w = resize_spectral(self.w_spectral, x.shape[-2], x.shape[-1])
        out = fft.irfft2((x.unsqueeze(1) * w).sum(dim=2), s=(H_in, H_in), norm='backward')
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out
