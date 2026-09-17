import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sp=json.load(open('split.json')); gr=json.load(open('grid.json'))
RES=[28,40,56]; TS=[150,300,500,700]
INK='#1b1b1f'; W='#c0392b'; C='#1f77b4'
plt.rcParams.update({'font.size':9,'axes.edgecolor':'#999','axes.labelcolor':INK,
                     'text.color':INK,'xtick.color':'#555','ytick.color':'#555',
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

fig,axes=plt.subplots(1,3,figsize=(12.5,3.7))

# --- panel A: retained-band error vs resolution
ax=axes[0]
for key,col,lab in [('white|white',W,'white-noise model'),('colored|colored',C,'band-limited-noise model')]:
    rows=sp[key]
    for t,ls,mk in zip(TS[:3],['-','--',':'],['o','s','^']):
        y=[r['rel_lo'] for r in rows if r['t']==t]
        ax.plot(RES,y,ls,marker=mk,ms=4,color=col,lw=1.6,alpha=0.9,
                label=f'{lab}, t={t}' if t==300 else None)
ax.axvline(28,color='#aaa',lw=0.8,ls=':')
ax.text(28.4,ax.get_ylim()[1]*0.96,'training res',fontsize=7.5,color='#777',va='top')
ax.set_xlabel('sampling resolution'); ax.set_ylabel('relative $\\epsilon$-error, retained band ($k\\leq14$)')
ax.set_title('A. Error grows in the band the model\nwas trained on',fontsize=9.5,loc='left')
ax.set_xticks(RES); ax.legend(fontsize=7,frameon=False)

# --- panel B: normalised growth
ax=axes[1]
for key,col,lab in [('white|white',W,'white noise'),('colored|colored',C,'band-limited noise')]:
    rows=sp[key]
    g=[]
    for t in TS:
        v=[r['rel_lo'] for r in rows if r['t']==t]
        g.append([vi/v[0] for vi in v])
    g=np.array(g).mean(axis=0)
    ax.plot(RES,g,marker='o',ms=5,color=col,lw=2,label=lab)
ax.axhline(1,color='#aaa',lw=0.8,ls=':')
ax.set_xlabel('sampling resolution'); ax.set_ylabel('error relative to training resolution')
ax.set_title('B. Band-limited noise makes the retained\nband resolution-independent',fontsize=9.5,loc='left')
ax.set_xticks(RES); ax.legend(fontsize=8,frameon=False)
for x,y in zip(RES,g): ax.annotate(f'{y:.2f}×',(x,y),textcoords='offset points',xytext=(6,-2),fontsize=7.5,color=C)

# --- panel C: recalibration does not explain it
ax=axes[2]
rows=gr['white|white']
xs,ys,th=[],[],[]
for R in (40,56):
    for t in TS:
        e=rows[f'{R}_{t}']
        xs.append(e['t']); ys.append(e['t_star']); th.append(e['t_theory'])
ax.plot([0,750],[0,750],color='#bbb',lw=1,ls='--',label='no recalibration needed')
ax.scatter(xs,ys,s=34,color=W,zorder=3,label='empirical optimum $t^*$')
ax.scatter(xs,th,s=34,facecolors='none',edgecolors='#888',zorder=3,label='SNR-shift theory')
ax.set_xlabel('true timestep $t$'); ax.set_ylabel('timestep fed to the model')
ax.set_title('C. The model does not want a different\nnoise level',fontsize=9.5,loc='left')
ax.legend(fontsize=7.5,frameon=False,loc='upper left')
fig.tight_layout()
fig.savefig('figure1.png',bbox_inches='tight',dpi=170)
print('saved figure1.png')
