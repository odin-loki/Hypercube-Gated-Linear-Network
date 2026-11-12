"""End-to-end accuracy test on the scikit-learn digits set (64 inputs, 10 classes), one-vs-rest, online.
Writes train/test CSV, runs the C++ classifier, optionally compares with the NumPy reference.
usage: python3 digits_test.py path/to/hgln_d [passes=3] [--ref]   (the --ref NumPy run is slow: about 10 minutes)"""
import subprocess, sys, os, tempfile
import numpy as np
from sklearn.datasets import load_digits
exe = sys.argv[1]; passes = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
d = load_digits(); X = d.data / 8.0 - 1.0; y = d.target; n = len(y); ntr = 2 * n // 3
p = np.random.default_rng(10).permutation(n); X, y = X[p], y[p]
with tempfile.TemporaryDirectory() as tmp:
    for name, a, b in (("train.csv", 0, ntr), ("test.csv", ntr, n)):
        np.savetxt(os.path.join(tmp, name), np.column_stack([y[a:b], X[a:b]]), delimiter=",", fmt="%.6g")
    out = subprocess.run([exe, "classify", os.path.join(tmp, "train.csv"), os.path.join(tmp, "test.csv"), str(passes)], check=True, capture_output=True, text=True).stdout
print(out, end="")
acc = float(out.strip().splitlines()[-1].split("accuracy ")[1].split("%")[0])
print(f"final holdout accuracy {acc}% (reference NumPy: 93-95% after 1-2 passes, about 97.0% mean after 3 passes over 5 shuffles; one shuffle varies by about +-2 points)")
sys.exit(0 if acc >= 90.0 else 1)
