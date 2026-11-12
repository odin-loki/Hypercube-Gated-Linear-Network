import numpy as np, json, sys, warnings
warnings.filterwarnings("ignore")
from tgn_v11 import NetT
from t7 import streamA, streamB, N, WIN

from t7 import DR
def streamC(seed):
    r=np.random.default_rng(9000+seed); X=r.uniform(-1,1,(N,8)); t=np.arange(N)
    def bumps(dims):
        C=r.uniform(-0.8,0.8,(4,3)); sg=np.array([1,-1,1,-1.])
        return lambda X:(sg[None]*np.exp(-((X[:,dims][:,None,:]-C[None])**2).sum(2)/(2*0.4**2))).sum(1)
    f1,f2=bumps([0,1,2]),bumps([0,3,4])
    return X,np.where(t<DR,f1(X),f2(X))+r.normal(0,0.05,N)
VAR={"D diversify (competence gain + conscience)":dict(diversify=True),
     "G global gradient (50% of error signal)":dict(glob=0.5),
     "D+G together":dict(diversify=True,glob=0.5)}
def syn(name,kw):
    for task,sf in (("A",streamA),("B",streamB),("C",streamC)):
        q=[]
        for sd in range(200,210):
            X,y=sf(sd); m=NetT(sd,8,**kw); err=np.zeros(N)
            for t in range(N):
                m.predict(X[t]); err[t]=(y[t]-m.cands()["competitive mix, all 48 units"])**2; m.learn(y[t])
            q.append([float(err[a:b].mean()/y[a:b].var()) for a,b in WIN])
        print(json.dumps({"kind":"syn","var":name,"task":task,"q":q}),flush=True)
def dig(name,kw):
    from sklearn.datasets import load_digits
    d=load_digits(); X=d.data/8.0-1.0; y=d.target; n=len(y); ntr=2*n//3; C=10; seed=10
    p=np.random.default_rng(seed).permutation(n); Xs,ys=X[p],y[p]; Xtr,ytr,Xte,yte=Xs[:ntr],ys[:ntr],Xs[ntr:],ys[ntr:]
    nets=[NetT(seed*100+k,64,16,g0=0.02,g1=0.02,**kw) for k in range(C)]; hs=[]
    for ep in range(1,4):
        for t in range(ntr):
            for nn in nets: nn.predict(Xtr[t])
            for k,nn in enumerate(nets): nn.learn(1.0 if k==ytr[t] else -1.0)
        ok=[]
        for x,lab in zip(Xte,yte):
            sc=[]
            for nn in nets: nn.predict(x); sc.append(nn.cands()["competitive mix, all 48 units"])
            ok.append(int(np.argmax(sc)==lab))
        hs.append(round(100*float(np.mean(ok)),1))
    print(json.dumps({"kind":"dig","var":name,"seed":seed,"holdout":hs}),flush=True)
if __name__=="__main__":
    for name,kw in VAR.items(): syn(name,kw); dig(name,kw)
