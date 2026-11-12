# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parity test: NumPy reference (hgln_ref.py) vs the C++ CLI, started from identical hyperplanes.
usage: python3 parity.py path/to/hgln_d|hgln_f [d] [tolerance] [v43|v42|v43f|v42f] [units]
  -> prints max |difference|, exits 1 above the tolerance.
v43 (default): plain-average mixture; v42: competence-weighted mixture; suffix f: fan-out (layer 3 reads layer 1).
v5: the version 5.0 float configuration (fan-out, cross-layer latch 0.1, latch credit 0.5, gates on the global error,
glob 0.95; all 16 corners); v5c: chain with latch 0.2, credit 1.0, gates on the units' own errors, glob 0.75.
units: units per layer (8, 12 or 16; default 16)."""
import subprocess, struct, sys, tempfile, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from hgln_ref import HGLN
exe = sys.argv[1]; d = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tol = float(sys.argv[3]) if len(sys.argv) > 3 else (1e-12 if exe.endswith("_d") else 1e-4)
mode = sys.argv[4] if len(sys.argv) > 4 else "v43"
units = int(sys.argv[5]) if len(sys.argv) > 5 else 16
if mode not in ("v43", "v42", "v43f", "v42f", "v5", "v5c"): sys.exit(f"unknown mode {mode}")
comp, fan = mode.startswith("v42"), mode.endswith("f") or mode == "v5"
latch, credit, gerr, glob = {"v5": (0.1, 0.5, 2, 0.95), "v5c": (0.2, 1.0, 1, 0.75)}.get(mode, (0.0, 0.0, 0, 0.75))
N = 2000; g0, g1 = (0.2, 0.05) if d <= 16 else (0.02, 0.02)
rng = np.random.default_rng(3)
X = rng.uniform(-1, 1, (N, d)); y = np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.5 * X[:, 2] * X[:, 3]
net = HGLN(11, d, Wd=units, glr0=g0, glr1=g1, competence=comp, fanout=fan, glob=glob, latch=latch, latch_credit=credit, gate_err=gerr)
with tempfile.TemporaryDirectory() as tmp:
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    with open(fin, "wb") as f:
        f.write(struct.pack("ii", d, N))
        for L in net.L: f.write(np.ascontiguousarray(L.H, dtype=np.float64).tobytes())   # (units, 4, d+1) each
        f.write(np.ascontiguousarray(X, dtype=np.float64).tobytes()); f.write(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    P = np.zeros(N)
    for t in range(N): P[t] = net.predict(X[t]); net.learn(y[t])
    subprocess.run([exe, "parity", fin, fout, str(g0), str(g1), "1" if comp else "0", str(units), "1" if fan else "0",
                    str(latch), str(credit), str(gerr), str(glob)], check=True)
    Q = np.fromfile(fout, dtype=np.float64)
diff = float(np.abs(P - Q).max())
print(f"parity {os.path.basename(exe)} {mode} d={d} units={units} N={N}: max |NumPy - C++| = {diff:.3g} (tolerance {tol:g})")
sys.exit(0 if diff <= tol else 1)
