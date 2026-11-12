# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import tgn_v5
from tgn_v5 import TGN5, HC
from tgn_final import feat

class HCG(HC):
    """Hybrid unit: one GLOBAL linear rule (always on) + 16 LOCAL corner rules weighted by the heatmap.
    output = global(x) + sum_c h_c * rule_c(x). Both parts train on the same error."""
    def __init__(s,U,n_in,K,d,rng,glr,balance=False,lam=0.0,gam=0.02):
        super().__init__(U,n_in,K,d,rng,glr,balance,lam,gam); s.Wg=np.zeros((U,n_in+1))
    def forward(s,a,u):
        super().forward(a,u)
        s.fg=(s.Wg*s.xm).sum(1); s.mix=s.mix+s.fg; s.pred=np.clip(s.mix,-2,2); return s.pred
    def update(s,y,mu,gscale=1.0):
        e=y-s.pred
        if s.glr>0:      # gate gradient: the global part is constant in the gates, so the local formula is unchanged
            hf=s.h*s.f; F1=hf@s.B; F0=hf.sum(1,keepdims=True)-F1
            g=tgn_v5.BETA*(F1*(1-s.a)-F0*s.a)
            c=np.clip(s.glr*gscale*e[:,None]*g,-0.05,0.05)
            dp=np.einsum('ub,ubd->ud',c,s.H*s.u[None,None,:])
            s.H=s.H+c[:,:,None]*s.pu[:,None,:]; s.H/=np.linalg.norm(s.H,axis=2,keepdims=True)
            s.p=np.clip(s.p+dp,0.0,3.0)
        c2=mu*e/(1e-3+(1.0+(s.h**2).sum(1))*(s.xm**2).sum(1))
        s.W+=(c2[:,None]*s.h)[:,:,None]*s.xm[:,None,:]; s.Wg+=c2[:,None]*s.xm
        s.use=0.99*s.use+0.01*s.h
        s.G=0.98*s.G+0.02*np.abs(e)[:,None]*np.abs(s.inp)[None]; s.A=0.98*s.A+0.02*np.abs(s.inp)[None]
        s.bu[s.bi]=s.u; s.be[s.bi]=np.abs(e); s.bi=(s.bi+1)%64
    def structure_step(s,g):
        M0=s.M.copy(); super().structure_step(g); s.Wg[(s.M>0)&(M0==0)]=0.0

class Net(TGN5):
    """Configurable net for diagnosis. mu_mode / bud_mode: 'surprise' (adaptive) or 'fixed' (no feedback from error)."""
    def __init__(s,seed,d,Wd=16,glr=0.2,mu=0.3,mu_mode="fixed",bud_mode="fixed",grow_prune=True,hybrid=False):
        r=np.random.default_rng(seed); Cls=HCG if hybrid else HC
        mk=lambda U,n: Cls(U,n,16,d,r,glr,False,0.0)
        s.L=[mk(Wd,d),mk(Wd,Wd),mk(Wd,Wd),mk(1,Wd)]
        s.warm=512; s.ef=s.es=0.5; s.t=0; s.g=1.0; s.mu,s.mm,s.bm,s.gp=mu,mu_mode,bud_mode,grow_prune
    def learn(s,y):
        e=abs(y-s.out); s.ef=0.95*s.ef+0.05*e; s.es=0.995*s.es+0.005*e
        s.g=float(np.clip(s.ef/(s.es+1e-6),0.5,3.0))
        mu=s.mu if s.mm=="fixed" else float(np.clip(s.mu*s.g,0.5*s.mu,2.7*s.mu))
        for L in s.L: L.update(y,mu,1.0)
        s.t+=1
        if s.gp and s.t>=s.warm and s.t%64==0:
            gb=1.0 if s.bm=="fixed" else s.g
            for L in s.L: L.structure_step(gb)
