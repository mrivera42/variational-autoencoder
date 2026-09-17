"""Where does off-resolution error live? Normalised spectral decomposition.

Key: fourier_resize puts EXACTLY ZERO signal energy above the training Nyquist
(k=14), so above k=14 the input x_t is pure noise. Recovering eps there is the
EASIEST possible task -- just rescale x_t. If the model fails there, that is a
band-limit/aliasing failure, not an information problem.
"""
import sys, json, numpy as np
sys.path.insert(0,'/home/claude/exp')
from fno_np import load_model, Diffusion
from arms import *

CKPT='/mnt/user-data/uploads/variational-autoencoder/checkpoints/'
MN='/mnt/user-data/uploads/variational-autoencoder/data/MNIST/raw/t10k-images-idx3-ubyte.gz'
D=Diffusion(); NYQ=14.0

def band_masks(R):
    fh=np.fft.fftfreq(R)*R
    k=np.sqrt(fh[:,None]**2+fh[None,:]**2)
    return k<=NYQ, k>NYQ

def analyse(model_name,kind,ts=(150,300,500,700),n=24):
    m=load_model(CKPT+f'ddpm_fno_{model_name}.pth')
    xs=load_mnist(MN,n=n)[:,None]
    rows=[]
    for R in (28,40,56):
        x0=fourier_resize(xs,R); lo,hi=band_masks(R)
        for t in ts:
            rng=np.random.default_rng(4242+t)
            eps=NOISE[kind]((n,1,R,R),rng)
            ab=D.ab[t]
            xt=(np.sqrt(ab)*x0+np.sqrt(1-ab)*eps).astype(np.float32)
            ehat=m(xt,np.full(n,t))
            E=np.fft.fft2(eps-ehat,norm='ortho'); N=np.fft.fft2(eps,norm='ortho')
            err_lo=(np.abs(E[...,lo])**2).sum(); err_hi=(np.abs(E[...,hi])**2).sum()
            nrg_lo=(np.abs(N[...,lo])**2).sum(); nrg_hi=(np.abs(N[...,hi])**2).sum()
            rows.append(dict(R=R,t=t,
                rel_lo=float(err_lo/max(nrg_lo,1e-12)),
                rel_hi=float(err_hi/max(nrg_hi,1e-12)) if hi.sum() else None,
                frac_above=float(err_hi/max(err_lo+err_hi,1e-12))))
    return rows

if __name__=='__main__':
    out={}
    for mn,kind in [('white','white'),('colored','colored')]:
        out[f'{mn}|{kind}']=analyse(mn,kind)
    json.dump(out,open('/home/claude/exp/split.json','w'),indent=1)
    for key,rows in out.items():
        print('='*70); print(key)
        print(f"{'R':>4} {'t':>5} | {'rel err BELOW nyq':>18} {'rel err ABOVE nyq':>18} {'% err above':>12}")
        for r in rows:
            hi = f"{r['rel_hi']:.4f}" if r['rel_hi'] is not None else "   n/a"
            print(f"{r['R']:>4} {r['t']:>5} | {r['rel_lo']:>18.4f} {hi:>18} {100*r['frac_above']:>11.1f}%")
