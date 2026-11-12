# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from tgn_v11 import NetT
from t7 import streamA, streamB, N, WIN
from t24 import streamC
for task,sf in (("A",streamA),("B",streamB),("C",streamC)):
    q=[]
    for sd in range(200,210):
        X,y=sf(sd); m=NetT(sd,8,glob=0.75,kappa=0.0); err=np.zeros(N)
        for t in range(N):
            m.predict(X[t]); o=np.concatenate([L.pred for L in m.L[:3]]); err[t]=(y[t]-o.mean())**2; m.learn(y[t])
        q.append([float(err[a:b].mean()/y[a:b].var()) for a,b in WIN])
    print(json.dumps({"task":task,"cfg":"uniform weights (no competition), share 0.75","q":q}),flush=True)
