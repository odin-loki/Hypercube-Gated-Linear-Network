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
