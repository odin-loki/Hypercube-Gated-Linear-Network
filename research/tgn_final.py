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


