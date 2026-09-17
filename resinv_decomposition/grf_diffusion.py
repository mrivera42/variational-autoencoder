"""
Noise-vs-aliasing decomposition for resolution-invariant diffusion.

Exact (training-free) testbed: Gaussian random fields with a known power-law
spectrum. For a Gaussian prior the optimal denoiser is the Wiener filter in
Fourier space, so we can build an EXACT diffusion model and isolate the two
candidate failure modes at off-training resolutions:

  (1) noise mis-specification : white noise per-mode power is 1/N^2, so it
      DEPENDS on the grid. Signal per-mode power S(k) does not. Changing N
      therefore changes per-mode SNR by r^2 even though the model is
      conditioned on an unchanged t.
  (2) aliasing : requires a nonlinearity. A pointwise nonlinearity creates
      harmonics above the grid Nyquist which fold back.

The linear arm CANNOT alias, so it measures (1) alone. That is the control the
literature is missing.
"""
import numpy as np

# ---------------------------------------------------------------- spectra

def wavenumbers(N):
    """Integer wavevector magnitudes for an N x N grid, continuum-indexed."""
    k1 = np.fft.fftfreq(N, d=1.0 / N)          # -N/2 .. N/2-1
    KX, KY = np.meshgrid(k1, k1, indexing="ij")
    return np.sqrt(KX**2 + KY**2), KX, KY

def spectrum(kmag, beta=3.0, k0=2.0):
    """Continuum power spectrum S(k). Power-law tail k^-beta, softened at k0."""
    return (1.0 + (kmag / k0) ** 2) ** (-beta / 2.0)

# ------------------------------------------------------- GRF sampling

def sample_grf(N, rng, beta=3.0, k0=2.0):
    """
    Draw a real Gaussian random field whose CONTINUUM per-mode power is S(k),
    independent of N. Convention: a = fft2(x)/N^2 approximates continuum coeffs.
    """
    kmag, _, _ = wavenumbers(N)
    S = spectrum(kmag, beta, k0)
    # complex white, Hermitian-symmetrised by taking the real part of the ifft
    w = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2.0)
    a = np.sqrt(S) * w
    x = np.fft.ifft2(a * N**2).real * np.sqrt(2.0)
    return x

def coeffs(x):
    """Continuum-normalised Fourier coefficients."""
    N = x.shape[-1]
    return np.fft.fft2(x) / N**2

def radial_power(x, nbins=None):
    """Radially averaged power spectrum, indexed by continuum wavenumber."""
    N = x.shape[-1]
    kmag, _, _ = wavenumbers(N)
    a = coeffs(x)
    P = np.abs(a) ** 2
    kflat, Pflat = kmag.ravel(), P.ravel()
    kmax = N // 2
    edges = np.arange(0.5, kmax + 1.5, 1.0)
    idx = np.digitize(kflat, edges)
    out_k, out_P = [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() > 0:
            out_k.append(kflat[m].mean())
            out_P.append(Pflat[m].mean())
    return np.array(out_k), np.array(out_P)

# ------------------------------------------------------- diffusion schedule

class VPSchedule:
    """Standard variance-preserving (DDPM) schedule in continuous time."""
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.bmin, self.bmax = beta_min, beta_max
    def log_alpha_bar(self, t):
        return -(0.5 * t**2 * (self.bmax - self.bmin) + t * self.bmin)
    def alpha_bar(self, t):
        return np.exp(self.log_alpha_bar(t))

# ------------------------------------------------------- the denoiser

class OperatorDenoiser:
    """
    An idealised neural-operator denoiser.

    Its parameters live in FREQUENCY space (like an FNO's spectral weights) and
    are fit at a training resolution N_train. At |k| <= k_max it applies the
    Wiener-optimal filter. Above k_max it either zeroes (band-limit, the FNO
    mode-truncation behaviour) or passes through (residual path).

    noise_power_train = per-mode power of the noise it was trained with.
    """
    def __init__(self, N_train, sched, beta=3.0, k0=2.0, k_max=None,
                 above="zero", nonlin=None, nonlin_gain=0.0):
        self.N_train = N_train
        self.sched = sched
        self.beta, self.k0 = beta, k0
        self.k_max = k_max if k_max is not None else N_train // 2
        self.above = above
        self.nonlin = nonlin
        self.nonlin_gain = nonlin_gain
        # what the model believes the per-mode noise power is at a given t
        self.noise_power_train = 1.0 / N_train**2

    def wiener(self, kmag, t):
        ab = self.sched.alpha_bar(t)
        S = spectrum(kmag, self.beta, self.k0)
        num = np.sqrt(ab) * S
        den = ab * S + (1.0 - ab) * self.noise_power_train
        return num / np.maximum(den, 1e-300)

    def predict_x0(self, x_t, t):
        N = x_t.shape[-1]
        kmag, _, _ = wavenumbers(N)
        H = self.wiener(kmag, t)
        band = kmag <= self.k_max
        a = coeffs(x_t)
        out = np.zeros_like(a)
        out[band] = H[band] * a[band]
        if self.above == "pass":
            out[~band] = a[~band]
        x0 = np.fft.ifft2(out * N**2).real
        if self.nonlin is not None and self.nonlin_gain > 0:
            # a pointwise nonlinearity: creates harmonics -> aliases on the grid
            x0 = x0 + self.nonlin_gain * self.nonlin(x0)
        return x0
