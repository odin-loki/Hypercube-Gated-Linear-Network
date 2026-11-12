# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parity test: NumPy reference (hgln_ref.py) vs the C++ CLI, started from identical hyperplanes.
usage: python3 parity.py path/to/hgln_d|hgln_f [d] [tolerance] [v43|v42|v43f|v42f|v5|v5d|v5c|v43k] [units]
  -> prints max |difference|, exits 1 above the tolerance.
v43 (default): plain-average mixture; v42: competence-weighted mixture; suffix f: fan-out (layer 3 reads layer 1).
v5: the version 5.0 configuration (fan-out, cross-layer latch 0.1, latch credit 0.5, gates on the global error,
glob 0.95, top-6 corners); v5d: the same with all 16 corners; v5c: chain with latch 0.2, credit 1.0, gates on the units'
own errors, glob 0.75, top-3 corners; v43k: version 4.3 with top-8 corners.
units: units per layer (8, 12 or 16; default 16).
With top-K corners (v5, v5c, v43k) the comparison allows 1e-4: a hard selection meets exact ties (saturated gates give
corners of exactly equal value), and NumPy's exp differs from the library's gate in the last bit, so one side can see a
tie the other breaks; the two then select a different near-zero corner and differ by about 1e-6. The library and the
GPU kernel share the gate function and agree exactly (cuda/cuda_emu.cpp). A float32 build (tolerance argument above
1e-6) meets near-ties at its own rounding, about 1e-7, far more often: it then follows a different, equally valid
trajectory than the float64 reference (float32 C++ diverges from float64 C++ just as much), so float32 with top-K is
not compared: it is refused here (exit 2). Its arithmetic is the float64 code's, checked by the float64 run."""
import subprocess, struct, sys, tempfile, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from hgln_ref import HGLN
exe = sys.argv[1]; d = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tol = float(sys.argv[3]) if len(sys.argv) > 3 else (1e-12 if exe.endswith("_d") else 1e-4)
mode = sys.argv[4] if len(sys.argv) > 4 else "v43"
units = int(sys.argv[5]) if len(sys.argv) > 5 else 16
if mode not in ("v43", "v42", "v43f", "v42f", "v5", "v5d", "v5c", "v43k"): sys.exit(f"unknown mode {mode}")
comp, fan = mode.startswith("v42"), mode.endswith("f") or mode in ("v5", "v5d")
latch, credit, gerr, glob = {"v5": (0.1, 0.5, 2, 0.95), "v5d": (0.1, 0.5, 2, 0.95), "v5c": (0.2, 1.0, 1, 0.75)}.get(mode, (0.0, 0.0, 0, 0.75))
corners = {"v5": 6, "v5c": 3, "v43k": 8}.get(mode, 16)
if corners < 16 and tol > 1e-6: sys.exit(f"parity: mode {mode} uses top-K corners; compare float64 builds only (see the docstring)")
N = 2000; g0, g1 = (0.2, 0.05) if d <= 16 else (0.02, 0.02)
rng = np.random.default_rng(3)
X = rng.uniform(-1, 1, (N, d)); y = np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.5 * X[:, 2] * X[:, 3]
net = HGLN(11, d, Wd=units, glr0=g0, glr1=g1, competence=comp, fanout=fan, glob=glob, latch=latch, latch_credit=credit, gate_err=gerr, corners=corners)
with tempfile.TemporaryDirectory() as tmp:
    fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    with open(fin, "wb") as f:
        f.write(struct.pack("ii", d, N))
        for L in net.L: f.write(np.ascontiguousarray(L.H, dtype=np.float64).tobytes())   # (units, 4, d+1) each
        f.write(np.ascontiguousarray(X, dtype=np.float64).tobytes()); f.write(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    P = np.zeros(N)
    for t in range(N): P[t] = net.predict(X[t]); net.learn(y[t])
    subprocess.run([exe, "parity", fin, fout, str(g0), str(g1), "1" if comp else "0", str(units), "1" if fan else "0",
                    str(latch), str(credit), str(gerr), str(glob), str(corners)], check=True)
    Q = np.fromfile(fout, dtype=np.float64)
dv = np.abs(P - Q); diff = float(dv.max())
note = ""
ok = diff <= tol
if corners < 16:
    first = int(np.argmax(dv > 1e-12)) if diff > 1e-12 else N
    note = f"; identical to 1e-12 up to sample {first} of {N}"; tol = max(tol, 1e-4); ok = diff <= tol
print(f"parity {os.path.basename(exe)} {mode} d={d} units={units} N={N}: max |NumPy - C++| = {diff:.3g} (tolerance {tol:g}{note})")
sys.exit(0 if ok else 1)
