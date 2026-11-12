# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the spec's benchmark data as raw float64 files for eval.cpp: synthetic streams A, B, C (research/t7.py, t24.py;
seeds 200-209, 4096 x 8, drift at 2048) and digits shuffles 10-14 (1198 train / 599 test, inputs scaled to [-1, 1])."""
import os, sys, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, ".."))
from t7 import streamA, streamB
from t24 import streamC
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "data"); os.makedirs(out, exist_ok=True)
for task, f in (("A", streamA), ("B", streamB), ("C", streamC)):
    for sd in range(200, 210):
        X, y = f(sd); np.ascontiguousarray(np.column_stack([X, y]), dtype=np.float64).tofile(os.path.join(out, f"syn_{task}_{sd}.bin"))
from sklearn.datasets import load_digits
d = load_digits(); X = d.data / 8.0 - 1.0; y = d.target.astype(np.float64); n = len(y); ntr = 2 * n // 3
for sh in range(10, 15):
    p = np.random.default_rng(sh).permutation(n)
    np.ascontiguousarray(np.column_stack([y[p], X[p]]), dtype=np.float64).tofile(os.path.join(out, f"dig_{sh}.bin"))
print("ok", out, ntr, n - ntr)
# ---- time-structured streams (version 4.4 study): 5 seeds each, 6000 samples; file = float64 [d, then N rows of d inputs + target]
def save_ts(name, sd, X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    a = np.concatenate([[X.shape[1]], np.column_stack([X, y]).ravel()])
    a.astype(np.float64).tofile(os.path.join(out, f"ts_{name}_{sd}.bin"))
NT = 6000
for sd in range(300, 305):
    r = np.random.default_rng(sd)
    # NARMA-10: input u(t) ~ U[0, 0.5]; target y(t+1). Model sees [u(t), y(t)] (y(t) is the label just learned).
    while True:   # NARMA-10 occasionally diverges; redraw the input sequence until it stays bounded (standard practice)
        u = r.uniform(0, 0.5, NT + 200); yy = np.zeros(NT + 201)
        with np.errstate(all="ignore"):
            for t in range(9, NT + 200):
                yy[t + 1] = 0.3 * yy[t] + 0.05 * yy[t] * yy[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
        if np.isfinite(yy).all() and np.abs(yy).max() < 1: break
    u, yy = u[200:], yy[200:]
    save_ts("narma10", sd, np.column_stack([4 * u - 1, 2 * yy[:NT] - 1]), 4 * yy[1:NT + 1] - 2)
    # Mackey-Glass (tau = 17): see s(t), predict s(t+1); the delayed term needs memory.
    dt = 0.1; tau = 17; n = int((NT + 400) / dt); hist = int(tau / dt); s = np.zeros(n); s[:hist + 1] = 1.2 + 0.1 * r.normal(size=hist + 1)
    for t in range(hist, n - 1):
        s[t + 1] = s[t] + dt * (0.2 * s[t - hist] / (1 + s[t - hist] ** 10) - 0.1 * s[t])
    m = s[::10][250:250 + NT + 1]; m = (m - m.mean()) / (2 * m.std())
    save_ts("mackey", sd, m[:NT, None], m[1:NT + 1])
    # Regime switching: hidden 2-state Markov regime (switch prob 0.01); y = f_regime(x); inputs x(t) (4), y(t-1), x(t-1)[:3].
    X = r.uniform(-1, 1, (NT, 4)); reg = np.zeros(NT, int)
    for t in range(1, NT): reg[t] = reg[t - 1] ^ (r.uniform() < 0.01)
    f0 = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]; f1 = np.cos(2 * X[:, 1]) - 0.5 * X[:, 0] * X[:, 3]
    y = np.where(reg == 0, f0, f1) + r.normal(0, 0.05, NT)
    prev_y = np.concatenate([[0], y[:-1]]); prev_x = np.vstack([np.zeros((1, 4)), X[:-1]])[:, :3]
    save_ts("regime", sd, np.column_stack([X, np.clip(prev_y / 2, -1, 1), prev_x]), y)
print("time series ok")
