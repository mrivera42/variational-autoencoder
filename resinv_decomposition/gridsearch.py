"""Empirical recalibration: search t_fed to minimise one-step error, then
compare the optimum against the analytic prediction snr(t') = (R/28)^2 snr(t)."""
import sys, json, numpy as np
sys.path.insert(0,'/home/claude/exp')
from fno_np import load_model, Diffusion
from arms import *

CKPT='/mnt/user-data/uploads/variational-autoencoder/checkpoints/'
MN='/mnt/user-data/uploads/variational-autoencoder/data/MNIST/raw/t10k-images-idx3-ubyte.gz'
D=Diffusion(); SNR=D.ab/(1.0-D.ab)

def theory_t(t,R,train=28):
    return int(np.argmin(np.abs(SNR-(R/train)**2*SNR[t])))

def err_at(model,x0,t_true,R,kind,t_fed,seed):
    rng=np.random.default_rng(seed)
    eps=NOISE[kind]((x0.shape[0],1,R,R),rng)
    ab=D.ab[t_true]
    xt=(np.sqrt(ab)*x0+np.sqrt(1-ab)*eps).astype(np.float32)
    ehat=model(xt,np.full(x0.shape[0],t_fed))
    err=ehat-eps
    b,a=radial_split(err,R)
    return float((err**2).mean()),float(b),float(a)

def run(model_name,kind,ts=(150,300,500,700),n=24):
    m=load_model(CKPT+f'ddpm_fno_{model_name}.pth')
    xs=load_mnist(MN,n=n)[:,None]
    out={}
    for R in (28,40,56):
        x0=fourier_resize(xs,R)
        for t in ts:
            seed=999+t
            base=err_at(m,x0,t,R,kind,t,seed)
            if R==28:
                out[f'{R}_{t}']=dict(R=R,t=t,t_star=t,t_theory=t,naive=base,best=base,curve=[])
                continue
            th=theory_t(t,R)
            lo=max(1,min(t,th)-90); hi=min(D.T-1,max(t,th)+90)
            cand=sorted(set(np.linspace(lo,hi,19).astype(int).tolist()+[t,th]))
            curve=[(int(c),err_at(m,x0,t,R,kind,int(c),seed)[0]) for c in cand]
            tstar=min(curve,key=lambda z:z[1])[0]
            out[f'{R}_{t}']=dict(R=R,t=t,t_star=int(tstar),t_theory=int(th),
                                 naive=base,best=err_at(m,x0,t,R,kind,int(tstar),seed),
                                 theory=err_at(m,x0,t,R,kind,int(th),seed),curve=curve)
            print(f"  R={R} t={t}: t*={tstar} theory={th} naive={base[0]:.4f} best={min(c[1] for c in curve):.4f}",flush=True)
    return out

if __name__=='__main__':
    res={}
    for mn,kind in [('white','white'),('colored','colored')]:
        print(f"== model={mn} noise={kind}",flush=True)
        res[f'{mn}|{kind}']=run(mn,kind)
    json.dump(res,open('/home/claude/exp/grid.json','w'),indent=1)
    print('DONE')
