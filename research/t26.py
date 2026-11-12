import numpy as np, json, sys, warnings
warnings.filterwarnings("ignore")
from tgn_v11 import NetT
import t24
from t24 import streamA, streamB, streamC, N, WIN
MIX="competitive mix, all 48 units"
if sys.argv[1]=="syn":
    for g in (0.75,1.0):
        for task,sf in (("A",streamA),("B",streamB),("C",streamC)):
            q=[]
            for sd in range(200,210):
                X,y=sf(sd); m=NetT(sd,8,glob=g); err=np.zeros(N)
                for t in range(N):
                    m.predict(X[t]); err[t]=(y[t]-m.cands()[MIX])**2; m.learn(y[t])
                q.append([float(err[a:b].mean()/y[a:b].var()) for a,b in WIN])
            print(json.dumps({"glob":g,"task":task,"q":q}),flush=True)
else:
    from sklearn.datasets import load_digits
    d=load_digits(); X=d.data/8.0-1.0; y=d.target; n=len(y); ntr=2*n//3; C=10
    for seed in range(10,15):
        p=np.random.default_rng(seed).permutation(n); Xs,ys=X[p],y[p]; Xtr,ytr,Xte,yte=Xs[:ntr],ys[:ntr],Xs[ntr:],ys[ntr:]
        for g in (1.0,):
            nets=[NetT(seed*100+k,64,16,g0=0.02,g1=0.02,glob=g) for k in range(C)]; hs=[]
            for ep in range(1,4):
                for t in range(ntr):
                    for nn in nets: nn.predict(Xtr[t])
                    for k,nn in enumerate(nets): nn.learn(1.0 if k==ytr[t] else -1.0)
                ok=[]
                for x,lab in zip(Xte,yte):
                    sc=[]
                    for nn in nets: nn.predict(x); sc.append(nn.cands()[MIX])
                    ok.append(int(np.argmax(sc)==lab))
                hs.append(round(100*float(np.mean(ok)),1))
            print(json.dumps({"glob":g,"seed":seed,"holdout":hs}),flush=True)
