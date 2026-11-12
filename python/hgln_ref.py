# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hypercube Gated Linear Network (HGLN), version 4.
Stateless online learner. Each unit has 4 learned hyperplanes -> a 16-corner heatmap -> a blend of 16 local linear rules.
The prediction is a competitive mixture over all hidden units (weights = each unit's local competence), and each unit
learns from a blend of its own error and its responsibility-share of the mixture's error (global gradient).
Fixed step size and connection budget (no error feedback into learning rates); gate learning rate on a fixed schedule.
Author: Odin Loch"""
import numpy as np
BETA = 12.0

def feat(x):
    v = np.append(x, 1.0)
    return v / np.linalg.norm(v)

class GatedLayer:
    """U units; each unit owns K gated linear rules (rows of W) and K gate codes (rows of E)."""
    def __init__(s, U, n_in, K, d, rng, glr):
        s.U, s.n, s.K, s.d, s.rng, s.glr = U, n_in, K, d, rng, glr
        p = rng.uniform(-1, 1, (U, K, d))
        v = np.concatenate([p, np.ones((U, K, 1))], 2)
        s.E = v / np.linalg.norm(v, axis=2, keepdims=True)      # gate codes (U,K,d+1)
        s.p = np.ones((U, d + 1))                                # learned per-input gate weights
        s.W = np.zeros((U, K, n_in + 1))                         # rule table; last column = bias (+1)
        s.M = np.ones((U, n_in + 1))                             # connection mask (dense at start)
        s.alive = np.ones((U, K), bool)                          # which codes exist
        s.use = np.full((U, K), 1.0 / K)                         # code usage (EMA)
        s.G = np.zeros((U, n_in + 1))                            # accumulated |error * input| (regrowth signal)
        s.A = np.full((U, n_in + 1), 1e-3)                       # input activity (EMA)
        s.bu = np.zeros((64, d + 1)); s.be = np.zeros((64, U)); s.bi = 0   # recent side-info / errors
        s.ar = np.arange(U)

    def forward(s, a, u):
        s.inp = np.append(a, 1.0); s.u = u; s.pu = s.p * u[None]
        sc = np.where(s.alive, np.einsum('ukd,ud->uk', s.E, s.pu), -1e9)
        s.idx = np.argpartition(-sc, 1, axis=1)[:, :2]           # top-2 codes per unit (sparse routing)
        top = np.take_along_axis(sc, s.idx, 1)
        z = np.exp(BETA * (top - top.max(1, keepdims=True)))
        s.sw = z / z.sum(1, keepdims=True)
        s.xm = s.inp[None] * s.M
        s.f = np.einsum('ukn,un->uk', s.W[s.ar[:, None], s.idx], s.xm)
        s.pred = np.clip((s.sw * s.f).sum(1), -2, 2)
        return s.pred

    def update(s, y, mu):
        e = y - s.pred
        if s.glr > 0:   # error-driven gate: gradient of the unit's own loss through the 2-way routing
            c = np.clip(s.glr * e * (s.f[:, 0] - s.f[:, 1]) * BETA * s.sw[:, 0] * s.sw[:, 1], -0.05, 0.05)
            j0, j1 = s.idx[:, 0], s.idx[:, 1]
            E0, E1 = s.E[s.ar, j0], s.E[s.ar, j1]
            s.p = np.clip(s.p + c[:, None] * ((E0 - E1) * s.u[None]), 0.0, 3.0)
            n0, n1 = E0 + c[:, None] * s.pu, E1 - c[:, None] * s.pu
            s.E[s.ar, j0] = n0 / np.linalg.norm(n0, axis=1, keepdims=True)
            s.E[s.ar, j1] = n1 / np.linalg.norm(n1, axis=1, keepdims=True)
        # convex local step on the two active rules (normalised LMS), no backprop
        c = mu * e / (1e-3 + (s.sw ** 2).sum(1) * (s.xm ** 2).sum(1))
        s.use *= 0.99
        for k in range(2):
            j = s.idx[:, k]
            s.W[s.ar, j] += (c * s.sw[:, k])[:, None] * s.xm
            s.use[s.ar, j] += 0.01 * s.sw[:, k]
        s.G = 0.98 * s.G + 0.02 * np.abs(e)[:, None] * np.abs(s.inp)[None]
        s.A = 0.98 * s.A + 0.02 * np.abs(s.inp)[None]
        s.bu[s.bi] = s.u; s.be[s.bi] = np.abs(e); s.bi = (s.bi + 1) % 64

    def structure_step(s, g):
        """Slow timescale. Surprise g sets the connection budget: calm -> prune, surprise -> regrow."""
        rho = float(np.clip(0.6 * g, 0.4, 1.0)); n = s.n
        nt = n if n <= 4 else int(round(rho * n))                # tiny input layers stay dense
        for u in range(s.U):
            act = np.where(s.M[u, :n] > 0)[0]; ina = np.where(s.M[u, :n] == 0)[0]
            if len(act) > nt:                                    # prune lowest |weight| x activity
                keep = np.abs(s.W[u][:, :n]).mean(0) * s.A[u, :n]
                s.M[u, act[np.argsort(keep[act])[:len(act) - nt]]] = 0
            elif len(act) < nt and len(ina):                     # regrow where error x input is largest
                gr = ina[np.argsort(-s.G[u, :n][ina])[:nt - len(act)]]
                s.M[u, gr] = 1; s.W[u][:, gr] = 0
            al = np.where(s.alive[u])[0]
            if g < 1.15 and len(al) > 4:                         # calm: drop a rarely used code
                j = al[np.argmin(s.use[u, al])]
                if s.use[u, j] < 0.35 / len(al): s.alive[u, j] = False
            elif g > 1.3:                                        # surprised: respawn a code where error was worst
                dead = np.where(~s.alive[u])[0]
                if len(dead):
                    uv = s.bu[np.argmax(s.be[:, u])]
                    w = al[np.argmax(s.E[u, al] @ (s.p[u] * uv))]
                    j = dead[0]; s.E[u, j] = uv / np.linalg.norm(uv)
                    s.W[u, j] = s.W[u, w]; s.alive[u, j] = True; s.use[u, j] = 1.0 / len(al)


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


class HC2(HC):
    """Hypercube unit with a per-corner competence map m (running squared error of the unit's own prediction).
    latch > 0 (version 5.0): cross-layer latch - with a lower layer given, each bit holds the same unit's bit of the
    lower layer inside the dead band |z| < latch: a = set + (1 - set - reset) a_lower."""
    def __init__(s, U, n_in, K, d, rng, glr, latch=0.0):
        super().__init__(U, n_in, K, d, rng, glr, False, 0.0); s.m = np.full((U, 16), 0.3); s.latch = latch
    def forward(s, a, u, lower=None):
        out = super().forward(a, u)
        s.latched = s.latch > 0 and lower is not None
        if s.latched:
            z = np.einsum('ubd,ud->ub', s.H, s.pu) + s.delta
            st = 1 / (1 + np.exp(-BETA * (z - s.latch))); rt = 1 / (1 + np.exp(-BETA * (-z - s.latch))); q0 = lower.a
            s.a = np.clip(st + (1 - st - rt) * q0, 1e-4, 1 - 1e-4)
            s.kap = st * (1 - st) * (1 - q0) + rt * (1 - rt) * q0; s.hold = 1 - st - rt
            s.h = np.prod(np.where(s.B[None] > 0.5, s.a[:, None, :], 1 - s.a[:, None, :]), axis=2)
            s.mix = (s.h * s.f).sum(1); s.pred = np.clip(s.mix, -2, 2); out = s.pred
        else:
            s.kap = s.a * (1 - s.a); s.hold = np.zeros_like(s.a)
        return out
    def comp(s):                                           # input-dependent competence of each unit
        return 1.0 / ((s.h * s.m).sum(1) + 1e-3)
    def update(s, y, mu, es, eg=None, gate_err=0, acc_in=None, acc_out=None, lower=None, credit=0.0):
        """es: the error the rules learn from; the gate steps use es (gate_err 0), the unit's own error (1) or the global
        error eg (2). Latch credit: the held part of each latched bit's gradient, times credit, goes to acc_out (the
        lower layer's bits); acc_in is added to this layer's (clipped) gate steps."""
        e = y - s.pred
        s.m = s.m + 0.02 * s.h * ((e ** 2)[:, None] - s.m)
        hf = s.h * s.f; F1 = hf @ s.B; F0 = hf.sum(1, keepdims=True) - F1
        ge = es if gate_err == 0 else (e if gate_err == 1 else np.full_like(es, eg))
        if s.latched:
            dm = F1 / s.a - F0 / (1 - s.a)
            c = np.clip(s.glr * ge[:, None] * (BETA * s.kap * dm), -0.05, 0.05)
            if credit > 0 and acc_out is not None:
                acc_out += np.clip(s.glr * ge[:, None] * (credit * BETA * lower.kap * dm * s.hold), -0.05, 0.05)
        else:
            g = BETA * (F1 * (1 - s.a) - F0 * s.a)         # d mix / d z_b
            c = np.clip(s.glr * ge[:, None] * g, -0.05, 0.05)
        if acc_in is not None: c = c + acc_in
        dp = np.einsum('ub,ubd->ud', c, s.H * s.u[None, None, :])
        s.H = s.H + c[:, :, None] * s.pu[:, None, :]; s.H /= np.linalg.norm(s.H, axis=2, keepdims=True)
        s.p = np.clip(s.p + dp, 0.0, 3.0)
        c2 = mu * es / (1e-3 + (s.h ** 2).sum(1) * (s.xm ** 2).sum(1))
        s.W += (c2[:, None] * s.h)[:, :, None] * s.xm[:, None, :]
        s.use = 0.99 * s.use + 0.01 * s.h
        s.G = 0.98 * s.G + 0.02 * np.abs(es)[:, None] * np.abs(s.inp)[None]
        s.A = 0.98 * s.A + 0.02 * np.abs(s.inp)[None]
        s.bu[s.bi] = s.u; s.be[s.bi] = np.abs(e); s.bi = (s.bi + 1) % 64

class HGLN:
    """3 hidden layers of Wd units (no separate readout). Prediction = competitive mixture over all 3*Wd units.
    fanout=True: layer 3 reads layer 1's outputs instead of layer 2's (layers 2 and 3 both fed by layer 1)."""
    def __init__(s, seed, d, Wd=16, glr0=0.2, glr1=0.05, tau=1000.0, mu=0.3, glob=0.75, warm=512, competence=True, fanout=False,
                 latch=0.0, latch_credit=0.0, gate_err=0):
        r = np.random.default_rng(seed)
        s.L = [HC2(Wd, d, 16, d, r, glr0, latch), HC2(Wd, Wd, 16, d, r, glr0, latch), HC2(Wd, Wd, 16, d, r, glr0, latch)]
        s.latch, s.credit, s.gate_err = latch, latch_credit, gate_err
        s.Wd, s.glr0, s.glr1, s.tau, s.mu, s.glob, s.warm, s.t = Wd, glr0, glr1, tau, mu, glob, warm, 0
        s.competence = competence   # False: version 4.3, plain average of the units (no competence weighting)
        s.fanout = fanout
    def predict(s, x):
        x = np.asarray(x, float); u = feat(x)
        a1 = s.L[0].forward(x, u); a2 = s.L[1].forward(a1, u, s.L[0])
        s.L[2].forward(a1 if s.fanout else a2, u, s.L[0] if s.fanout else s.L[1])
        s.o = np.concatenate([L.pred for L in s.L])
        if s.competence:
            comp = np.concatenate([L.comp() for L in s.L])
            s.r = comp / comp.sum()                        # competitive weights over units
        else:
            s.r = np.full(len(s.o), 1.0 / len(s.o))        # version 4.3: equal weights
        s.out = float((s.r * s.o).sum()); return s.out
    def learn(s, y):
        gl = s.glr1 + (s.glr0 - s.glr1) * np.exp(-s.t / s.tau)          # fixed gate-rate schedule
        for L in s.L: L.glr = gl
        e_loc = y - s.o; e_glob = y - s.out
        share = s.r / (s.r ** 2).sum() if s.competence else 1.0             # equal weights: r / sum r^2 = 1
        es = (1 - s.glob) * e_loc + s.glob * e_glob * share   # own error + responsibility-share of mixture error
        if s.latch > 0 and s.credit > 0:                                  # layers 3, 2, 1: credit reaches layer 1
            acc = [np.zeros((s.Wd, 4)) for _ in s.L]
            for i in (2, 1, 0):
                lo = None if i == 0 else (0 if s.fanout else i - 1)
                s.L[i].update(y, s.mu, es[i * s.Wd:(i + 1) * s.Wd], e_glob, s.gate_err, acc[i], None if lo is None else acc[lo],
                              None if lo is None else s.L[lo], s.credit)
        else:
            for i, L in enumerate(s.L): L.update(y, s.mu, es[i * s.Wd:(i + 1) * s.Wd], e_glob, s.gate_err)
        s.t += 1
        if s.t >= s.warm and s.t % 64 == 0:                              # consolidation: fixed density 0.6
            for L in s.L: L.structure_step(1.0)
