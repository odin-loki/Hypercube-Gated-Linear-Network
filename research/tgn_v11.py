# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import tgn_v5
from tgn_v10 import HCS, NetS

class HCT(HCS):
    """HCS whose update accepts a separate learning-error signal and a per-unit gain (competition / global gradient hooks)."""
    def update(s,y,mu,gscale=1.0,e_sig=None,gain=None):
        e=y-s.pred                                        # own error: used for the competence map
        s.m=s.m+0.02*s.h*((e**2)[:,None]-s.m)
        es=e if e_sig is None else e_sig; gn=1.0 if gain is None else gain
        if s.glr>0:
            hf=s.h*s.f; F1=hf@s.B; F0=hf.sum(1,keepdims=True)-F1
            g=tgn_v5.BETA*(F1*(1-s.a)-F0*s.a)
            c=np.clip(s.glr*gscale*(gn*es)[:,None]*g,-0.05,0.05)
            dp=np.einsum('ub,ubd->ud',c,s.H*s.u[None,None,:])
            s.H=s.H+c[:,:,None]*s.pu[:,None,:]; s.H/=np.linalg.norm(s.H,axis=2,keepdims=True)
            s.p=np.clip(s.p+dp,0.0,3.0)
        c2=(mu*gn)*es/(1e-3+(s.h**2).sum(1)*(s.xm**2).sum(1))
        s.W+=(c2[:,None]*s.h)[:,:,None]*s.xm[:,None,:]
        s.use=0.99*s.use+0.01*s.h
        s.G=0.98*s.G+0.02*np.abs(es)[:,None]*np.abs(s.inp)[None]; s.A=0.98*s.A+0.02*np.abs(s.inp)[None]
        s.bu[s.bi]=s.u; s.be[s.bi]=np.abs(e); s.bi=(s.bi+1)%64

class NetT(NetS):
    """diversify: units learn faster where they are relatively more competent than their own average (conscience-balanced competition).
    glob: share of each unit's error signal that comes from the ensemble's error, split by responsibility (global gradient of the mixture)."""
    def __init__(s,seed,d,Wd=16,g0=0.2,g1=0.05,tau=1000.0,mu=0.3,diversify=False,glob=0.0,kappa=1.0):
        r=np.random.default_rng(seed); mk=lambda U,n: HCT(U,n,16,d,r,g0,False,0.0)
        s.L=[mk(Wd,d),mk(Wd,Wd),mk(Wd,Wd),mk(1,Wd)]
        s.warm=512; s.t=0; s.mu=mu; s.g0,s.g1,s.tau=g0,g1,tau; s.div,s.glob,s.kappa,s.Wd=diversify,glob,kappa,Wd
        s.ubar=np.full(3*Wd,1.0/(3*Wd))
    def learn(s,y):
        gl=s.g1+(s.g0-s.g1)*np.exp(-s.t/s.tau)
        for L in s.L: L.glr=gl
        hid=s.L[:3]; Wd=s.Wd
        o=np.concatenate([L.pred for L in hid]); comp=np.concatenate([L.comp() for L in hid])**s.kappa
        r=comp/comp.sum(); eg=y-float((r*o).sum()); e_loc=y-o
        e_sig=e_loc.copy(); gain=np.ones_like(o)
        if s.glob>0: e_sig=(1-s.glob)*e_loc+s.glob*eg*r/(r**2).sum()     # each unit moves by its responsibility-share of the ensemble error
        if s.div:
            s.ubar=0.99*s.ubar+0.01*r; gain=np.clip(r/s.ubar,0.25,2.5)
        for i,L in enumerate(hid):
            sl=slice(i*Wd,(i+1)*Wd); L.update(y,s.mu,1.0,e_sig=e_sig[sl],gain=gain[sl])
        s.L[3].update(y,s.mu,1.0); s.t+=1
        if s.t>=s.warm and s.t%64==0:
            for L in s.L: L.structure_step(1.0)
