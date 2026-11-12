import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from tgn_v11 import NetT
from t7 import streamA
res={}
for g in (0.0,0.5,0.75,1.0):
    rows=[]
    for sd in (200,201,202):
        X,y=streamA(sd); m=NetT(sd,8,glob=g)
        for t in range(1536): m.predict(X[t]); m.learn(y[t])
        O=[[] for _ in range(3)]; Rr=[]; ens=[]; neff=[]
        for t in range(1536,2036):
            m.predict(X[t]); [O[i].append(m.L[i].pred.copy()) for i in range(3)]
            comp=np.concatenate([L.comp() for L in m.L[:3]]); r=comp/comp.sum(); neff.append(1.0/(r**2).sum())
            Rr.append(float((r*np.concatenate([L.pred for L in m.L[:3]])).sum()))
        yt=y[1536:2036]; v=yt.var(); row={"nmse_mix":float(((np.array(Rr)-yt)**2).mean()/v),"neff_units":float(np.mean(neff))}
        for i in range(3):
            A=np.array(O[i]); C=np.corrcoef(A.T); off=float(np.nanmean(np.abs(C[~np.eye(16,dtype=bool)])))
            ev=np.clip(np.linalg.eigvalsh(np.cov(A.T)),0,None); pr=float(ev.sum()**2/(ev**2).sum())
            row[f"corr{i+1}"]=off; row[f"rank{i+1}"]=pr; row[f"unit_nmse{i+1}"]=float(np.mean([((A[:,u]-yt)**2).mean()/v for u in range(16)]))
        rows.append(row)
    res[g]={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(json.dumps({"glob":g,**res[g]}),flush=True)
