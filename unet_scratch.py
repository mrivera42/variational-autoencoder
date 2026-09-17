"""A very basic U-Net from scratch, for intuition (plain CNN layers, no DDPM).

Build this yourself; the docstrings fully specify the target so you can check
your work. Ping me when you want to compare. Once this clicks, a UNO is the same
skeleton with SpectralConv2d in place of Conv2d and spectral (de)resampling in
place of MaxPool/ConvTranspose.

------------------------------------------------------------------------------
THE IDEA (what a U-Net is)
------------------------------------------------------------------------------
A U-Net has three parts, drawn as a "U":

    contracting path (encoder)              expanding path (decoder)
    high res, few channels                  high res, few channels
        |  each step: halve H,W                  ^  each step: double H,W
        |             double channels            |             halve channels
        v                                        |
              bottleneck (low res, many channels)

The trick that makes it a "U" and not just an autoencoder: SKIP CONNECTIONS.
Each encoder stage's feature map (before it is downsampled) is saved and later
CONCATENATED (along the channel axis) onto the matching-resolution decoder stage.
This hands fine spatial detail directly across the U, so the decoder doesn't have
to reconstruct it from the compressed bottleneck alone.

------------------------------------------------------------------------------
SHAPE WALKTHROUGH  (input 1 x 32 x 32, base=32, depth 2 -- your target)
------------------------------------------------------------------------------
    x            : [B,  1, 32, 32]
    enc1 = DoubleConv(1  -> 32) (x)      -> [B, 32, 32, 32]   <-- save as skip1
    p1   = MaxPool2d(2)(enc1)            -> [B, 32, 16, 16]
    enc2 = DoubleConv(32 -> 64)(p1)      -> [B, 64, 16, 16]   <-- save as skip2
    p2   = MaxPool2d(2)(enc2)            -> [B, 64,  8,  8]
    bott = DoubleConv(64 -> 128)(p2)     -> [B,128,  8,  8]    (bottleneck)

    u2   = ConvTranspose2d(128->64)(bott)-> [B, 64, 16, 16]
    c2   = cat([u2, skip2], dim=1)       -> [B,128, 16, 16]   (64 + 64)
    dec2 = DoubleConv(128 -> 64)(c2)     -> [B, 64, 16, 16]
    u1   = ConvTranspose2d(64 ->32)(dec2)-> [B, 32, 32, 32]
    c1   = cat([u1, skip1], dim=1)       -> [B, 64, 32, 32]   (32 + 32)
    dec1 = DoubleConv(64 -> 32)(c1)      -> [B, 32, 32, 32]

    out  = Conv2d(32 -> out_ch, 1x1)(dec1) -> [B, out_ch, 32, 32]
------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """The basic U-Net building block: two 3x3 convs, each followed by a
    nonlinearity, mapping in_ch -> out_ch and keeping H,W the same.

    Use padding=1 on the 3x3 convs so spatial size is preserved (only the
    channel count changes here; MaxPool/ConvTranspose do the resizing). A ReLU
    after each conv is enough for a basic version. (BatchNorm optional -- skip
    it for now to keep the intuition clean.)

    forward(x): [B, in_ch, H, W] -> [B, out_ch, H, W]
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # TODO: build a small nn.Sequential: conv3x3(in->out) -> ReLU
        #       -> conv3x3(out->out) -> ReLU
        raise NotImplementedError

    def forward(self, x):
        # TODO: run x through the block and return it
        raise NotImplementedError


class UNet(nn.Module):
    """A minimal 2-level U-Net (see the shape walkthrough in the module docstring).

    Args:
        in_ch:  input channels (1 for MNIST).
        out_ch: output channels (1 to reconstruct/segment a single map).
        base:   channel width of the first stage (doubles each level down).

    Pieces you need to create in __init__ (names are just suggestions):
        enc1 : DoubleConv(in_ch -> base)
        enc2 : DoubleConv(base  -> base*2)
        pool : nn.MaxPool2d(2)                 (can reuse one instance)
        bott : DoubleConv(base*2 -> base*4)    (bottleneck)
        up2  : nn.ConvTranspose2d(base*4 -> base*2, kernel_size=2, stride=2)
        dec2 : DoubleConv(base*4 -> base*2)    (base*4 in, because of concat!)
        up1  : nn.ConvTranspose2d(base*2 -> base, kernel_size=2, stride=2)
        dec1 : DoubleConv(base*2 -> base)      (base*2 in, because of concat!)
        outc : nn.Conv2d(base -> out_ch, kernel_size=1)

    The two "gotchas" to get right:
      1. SAVE each encoder output BEFORE pooling -- those are your skips.
      2. After each up-conv, torch.cat([upsampled, matching_skip], dim=1)
         BEFORE the decoder DoubleConv. That concat is why dec2/dec1 take
         DOUBLE the channels as input.

    forward(x): [B, in_ch, H, W] -> [B, out_ch, H, W]  (H, W unchanged overall)
    """

    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        # TODO: create the layers listed above
        raise NotImplementedError

    def forward(self, x):
        # TODO: encoder (save skips) -> bottleneck -> decoder (up, concat skip,
        #       DoubleConv) -> 1x1 output conv. Follow the shape walkthrough.
        raise NotImplementedError


if __name__ == "__main__":
    # Smoke test: fill in the class, then run `python unet_scratch.py`.
    # A correct build prints:  torch.Size([2, 1, 32, 32])
    net = UNet(in_ch=1, out_ch=1, base=32)
    x = torch.randn(2, 1, 32, 32)
    print(net(x).shape)
