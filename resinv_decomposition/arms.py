"""Noise arms + data utilities for the resolution decomposition experiment."""
import numpy as np, gzip

TRAIN_RES = 28
CUTOFF = 8.0   # cycles-per-image, matches train_resinv_overnight.py

def load_mnist(path, n=256, seed=0):
    with gzip.open(path, 'rb') as f:
        buf = f.read()
    n_img = int.from_bytes(buf[4:8], 'big')
    imgs = np.frombuffer(buf[16:], dtype=np.uint8).reshape(n_img, 28, 28)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_img, n, replace=False)
    x = imgs[idx].astype(np.float32) / 255.0
    return (x - 0.5) / 0.5            # [-1,1], matches their Normalize((0.5,),(0.5,))

def fourier_resize(x, R):
    """Band-limited resample of the SAME continuum function onto an R x R grid.
    Zero-pad (up) or truncate (down) the spectrum. Preserves continuum amplitude."""
    B, C, H, W = x.shape
    X = np.fft.fftshift(np.fft.fft2(x, norm='ortho'), axes=(-2, -1))
    out = np.zeros((B, C, R, R), dtype=complex)
    m = min(H, R)
    hs, ho = (H - m) // 2, (R - m) // 2
    out[..., ho:ho+m, ho:ho+m] = X[..., hs:hs+m, hs:hs+m]
    y = np.fft.ifft2(np.fft.ifftshift(out, axes=(-2, -1)), norm='ortho').real
    return (y * (R / H)).astype(np.float32)     # ortho norm -> amplitude scale

def white_noise(shape, rng):
    return rng.normal(size=shape).astype(np.float32)

def band_limited_noise(shape, rng, cutoff=CUTOFF):
    """Verbatim port of train_resinv_overnight.band_limited_noise."""
    B, C, H, W = shape
    white = rng.normal(size=shape).astype(np.float32)
    fh = np.fft.fftfreq(H) * H
    fw = np.fft.rfftfreq(W) * W
    r = np.sqrt(fh[:, None]**2 + fw[None, :]**2)
    env = np.exp(-(r / cutoff)**2)
    n = np.fft.irfft2(np.fft.rfft2(white) * env, s=(H, W))
    sd = n.std(axis=(1, 2, 3), keepdims=True)
    return (n / (sd + 1e-8)).astype(np.float32)

NOISE = {'white': white_noise, 'colored': band_limited_noise}

# --------------------------------------------------------------- metrics
def radial_split(err, R, train_res=TRAIN_RES):
    """Split a per-pixel error field's spectral energy at the TRAINING Nyquist.
    Returns (energy_below, energy_above)."""
    E = np.abs(np.fft.fft2(err, norm='ortho'))**2
    fh = np.fft.fftfreq(R) * R
    k = np.sqrt(fh[:, None]**2 + fh[None, :]**2)
    nyq_train = train_res / 2.0
    below = E[..., k <= nyq_train].sum(axis=-1)
    above = E[..., k > nyq_train].sum(axis=-1)
    return below.mean(), above.mean()

def radial_power(x):
    """Radially averaged power spectrum indexed by cycles-per-image."""
    R = x.shape[-1]
    X = np.fft.fft2(x, norm='ortho')
    P = (np.abs(X)**2).mean(axis=(0, 1))
    fh = np.fft.fftfreq(R) * R
    k = np.sqrt(fh[:, None]**2 + fh[None, :]**2)
    kmax = R // 2
    edges = np.arange(0.5, kmax + 1.5)
    idx = np.digitize(k.ravel(), edges)
    ks, Ps = [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum():
            ks.append(k.ravel()[m].mean()); Ps.append(P.ravel()[m].mean())
    return np.array(ks), np.array(Ps)
