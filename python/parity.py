# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parity test: NumPy reference (hgln_ref.py) vs the C++ CLI, started from identical hyperplanes.
usage: python3 parity.py path/to/hgln_d|hgln_f  [d]  -> prints max |difference| and exits 1 if above the tolerance."""
import subprocess, struct, sys, tempfile, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from hgln_ref import HGLN
exe = sys.argv[1]; d = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tol = 1e-12 if exe.endswith("_d") else 1e-4
N = 2000; g0, g1 = (0.2, 0.05) if d <= 16 else (0.02, 0.02)
rng = np.random.default_rng(3)
X = rng.uniform(-1, 1, (N, d)); y = np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.5 * X[:, 2] * X[:, 3]
net = HGLN(11, d, glr0=g0, glr1=g1)
with tempfile.TemporaryDirectory() as tmp:
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    with open(fin, "wb") as f:
        f.write(struct.pack("ii", d, N))
        for L in net.L: f.write(np.ascontiguousarray(L.H, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(X, dtype=np.float64).tobytes()); f.write(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    P = np.zeros(N)
    for t in range(N): P[t] = net.predict(X[t]); net.learn(y[t])
    subprocess.run([exe, "parity", fin, fout, str(g0), str(g1)], check=True)
    Q = np.fromfile(fout, dtype=np.float64)
diff = float(np.abs(P - Q).max())
print(f"parity {os.path.basename(exe)} d={d} N={N}: max |NumPy - C++| = {diff:.3g} (tolerance {tol:g})")
sys.exit(0 if diff <= tol else 1)
