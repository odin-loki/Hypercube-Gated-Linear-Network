import numpy as np
from tgn_final import GatedLayer, feat, BETA

class HC(GatedLayer):
    """Hypercube from 4 learned hyperplanes: corner c (4 bits) gets weight prod_b sigma(+-beta*z_b).
    The 16 corner weights are the heatmap over the tesseract (they sum to 1). Neighbours = one bit flipped.
    MoE-style load balancing: each hyperplane's offset is nudged so it splits the data about 50/50."""
    def __init__(s,U,n_in,K,d,rng,glr,balance=True,lam=0.0,gam=0.02):
        super().__init__(U,n_in,16,d,rng,glr)
        h=rng.normal(size=(U,4,d+1)); h[:,:,-1]=0.0
        s.H=h/np.linalg.norm(h,axis=2,keepdims=True)
        s.B=((np.arange(16)[:,None]>>np.arange(4)[None])&1).astype(float)
        s.delta=np.zeros((U,4)); s.mb=np.full((U,4),0.5); s.balance,s.lam,s.gam=balance,lam,gam
    def forward(s,a,u):
        s.inp=np.append(a,1.0); s.u=u; s.pu=s.p*u[None]
        z=np.einsum('ubd,ud->ub',s.H,s.pu)+s.delta
        s.a=np.clip(1/(1+np.exp(-BETA*z)),1e-4,1-1e-4)
        s.h=np.prod(np.where(s.B[None]>0.5,s.a[:,None,:],1-s.a[:,None,:]),axis=2)
        s.xm=s.inp[None]*s.M
        s.f=np.einsum('ucn,un->uc',s.W,s.xm)
        s.mix=(s.h*s.f).sum(1); s.pred=np.clip(s.mix,-2,2); return s.pred
    def update(s,y,mu,gscale=1.0):
        e=y-s.pred
        if s.glr>0:
            hf=s.h*s.f; F1=hf@s.B; F0=hf.sum(1,keepdims=True)-F1
            g=BETA*(F1*(1-s.a)-F0*s.a)                      # d mix / d z_b
            c=np.clip(s.glr*gscale*e[:,None]*g,-0.05,0.05)
            dp=np.einsum('ub,ubd->ud',c,s.H*s.u[None,None,:])
            s.H=s.H+c[:,:,None]*s.pu[:,None,:]
            s.H/=np.linalg.norm(s.H,axis=2,keepdims=True)
            s.p=np.clip(s.p+dp,0.0,3.0)
        s.mb=0.99*s.mb+0.01*s.a
        if s.balance: s.delta-=s.gam*(s.mb-0.5)               # load balancing, no auxiliary loss
        c2=mu*e/(1e-3+(s.h**2).sum(1)*(s.xm**2).sum(1))
        s.W+=(c2[:,None]*s.h)[:,:,None]*s.xm[:,None,:]
        s.use=0.99*s.use+0.01*s.h
        s.G=0.98*s.G+0.02*np.abs(e)[:,None]*np.abs(s.inp)[None]
        s.A=0.98*s.A+0.02*np.abs(s.inp)[None]
        s.bu[s.bi]=s.u; s.be[s.bi]=np.abs(e); s.bi=(s.bi+1)%64
    def structure_step(s,g):
        super().structure_step(g); s.alive[:]=True
        if s.lam>0:                                           # light smoothing of RULES across cube edges
            Wn=np.mean([s.W[:,np.arange(16)^(1<<b)] for b in range(4)],axis=0)
            s.W=(1-s.lam)*s.W+s.lam*Wn

class TGN5:
    def __init__(s,seed,d,balance=True,lam=0.0,glr=0.2,Wd=16):
        r=np.random.default_rng(seed)
        mk=lambda U,n: HC(U,n,16,d,r,glr,balance,lam)
        s.L=[mk(Wd,d),mk(Wd,Wd),mk(Wd,Wd),mk(1,Wd)]
        s.warm=512; s.ef=s.es=0.5; s.t=0; s.g=1.0
    def predict(s,x):
        x=np.asarray(x,float); u=feat(x); a=x
        for L in s.L: a=L.forward(a,u)
        s.out=a[0]; return s.out
    def learn(s,y):
        e=abs(y-s.out); s.ef=0.95*s.ef+0.05*e; s.es=0.995*s.es+0.005*e
        s.g=float(np.clip(s.ef/(s.es+1e-6),0.5,3.0)); mu=float(np.clip(0.3*s.g,0.15,0.8))
        for L in s.L: L.update(y,mu,1.0)
        s.t+=1
        if s.t>=s.warm and s.t%64==0:
            for L in s.L: L.structure_step(s.g)
    def stats(s):   # mask density, corner-usage entropy (1 = all 16 corners used evenly, 0 = collapsed), bit imbalance
        ent=[]
        for L in s.L:
            q=L.use/np.maximum(L.use.sum(1,keepdims=True),1e-9); ent.append((-(q*np.log(q+1e-12)).sum(1)/np.log(16)).mean())
        return [float(np.mean([L.M[:,:L.n].mean() for L in s.L])), float(np.mean(ent)), float(np.mean([np.abs(L.mb-0.5).mean() for L in s.L]))]
    def pvec(s): return np.mean([L.p[:,:8].mean(0) for L in s.L[:3]],0)
