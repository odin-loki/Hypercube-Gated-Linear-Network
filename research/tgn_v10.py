import numpy as np
from tgn_v5 import HC
from tgn_v8 import Net

class HCS(HC):
    """HC + per-corner running squared error of this unit's own prediction (its 'competence map')."""
    def __init__(s,U,n_in,K,d,rng,glr,balance=False,lam=0.0,gam=0.02):
        super().__init__(U,n_in,K,d,rng,glr,balance,lam,gam); s.m=np.full((U,16),0.3)
    def update(s,y,mu,gscale=1.0):
        e=y-s.pred
        s.m=s.m+0.02*s.h*((e**2)[:,None]-s.m)
        super().update(y,mu,gscale)
    def comp(s): return 1.0/((s.h*s.m).sum(1)+1e-3)       # input-dependent competence of each unit (inverse expected squared error)

class NetS(Net):
    """Loop-free HGLN with competence tracking and annealed gate rate; exposes competitive global mixtures over units."""
    def __init__(s,seed,d,Wd=16,g0=0.2,g1=0.05,tau=1000.0,mu=0.3):
        r=np.random.default_rng(seed); mk=lambda U,n: HCS(U,n,16,d,r,g0,False,0.0)
        s.L=[mk(Wd,d),mk(Wd,Wd),mk(Wd,Wd),mk(1,Wd)]
        s.warm=512; s.ef=s.es=0.5; s.t=0; s.g=1.0; s.mu,s.mm,s.bm,s.gp=mu,"fixed","fixed",True; s.g0,s.g1,s.tau=g0,g1,tau
    def learn(s,y):
        gl=s.g1+(s.g0-s.g1)*np.exp(-s.t/s.tau)
        for L in s.L: L.glr=gl
        super().learn(y)
    def mix(s,layers,kappa):
        o=np.concatenate([L.pred for L in layers]); w=np.concatenate([L.comp() for L in layers])**kappa
        return float((w*o).sum()/w.sum())
    def cands(s):
        R=s.out; H=s.L[:3]; M3=s.mix(H[2:],1.0); MA=s.mix(H,1.0)
        return {"readout (current)":R,"competitive mix, last layer":M3,"competitive mix, all 48 units":MA,
                "average of readout and all-unit mix":0.5*(R+MA),"competitive mix, all units, sharper (k=2)":s.mix(H,2.0)}
