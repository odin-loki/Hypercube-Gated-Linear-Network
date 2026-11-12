# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coordinate-descent tuning of the v5 additions. Tunes on seed sets 0 and 1 (paired with the v4.3 rows baseline/noise_1),
reports the final choice also on seed set 2, which tuning never sees. Score: mean relative NMSE change over the 6
synthetic windows and 3 time series (late windows), minus 5% per digits point gained; lower is better."""
import json, subprocess, sys
FIXED = "uniform=1 norm_track=1 U=8 hybrid=3 theta_mode=4 gate_err=2 credit=1".split()
GRID = {"theta": [0.05, 0.075, 0.1, 0.15], "glob": [0.85, 0.9, 0.95, 1.0], "sparse_k": [6, 8, 10, 16],
        "glr_scale": [0.5, 1, 1.5, 2], "mu": [0.25, 0.3, 0.4], "credit_scale": [0.5, 1, 2, 4]}
cur = {"theta": 0.1, "glob": 0.95, "sparse_k": 8, "glr_scale": 1, "mu": 0.3, "credit_scale": 1}
R = {json.loads(l)["name"]: json.loads(l) for l in open("results45.jsonl")}
BASE = {0: R["baseline"], 1: R["noise_1"], 2: R["noise_2"]}
cache = {}
def run(cfg, seed):
    key = (tuple(sorted(cfg.items())), seed)
    if key in cache: return cache[key]
    args = ["./eval", "data", "t"] + FIXED + [f"{k}={v}" for k, v in cfg.items()] + [f"seed_off={seed}"]
    r = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    cache[key] = r; return r
def rel(r, b):
    v = []
    for t in "ABC":
        for w in (1, 3): v.append((r["syn"][t]["mean"][w] - b["syn"][t]["mean"][w]) / b["syn"][t]["mean"][w])
    for t in ("narma10", "mackey", "regime"): v.append((r["ts"][t]["mean"][1] - b["ts"][t]["mean"][1]) / b["ts"][t]["mean"][1])
    return v, r["digits"]["mean"][2] - b["digits"]["mean"][2]
def score(cfg, seeds=(0, 1)):
    s = []
    for sd in seeds:
        v, dg = rel(run(cfg, sd), BASE[sd]); s.append(sum(v) / len(v) - 0.05 * dg)
    return sum(s) / len(s)
best = score(cur); print("start", cur, f"{best*100:+.2f}%", flush=True)
for rnd in range(2):
    changed = False
    for k, vals in GRID.items():
        for v in vals:
            if v == cur[k]: continue
            c = dict(cur); c[k] = v; sc = score(c)
            print(f"  round {rnd} {k}={v}: {sc*100:+.2f}%", flush=True)
            if sc < best - 0.002: best, cur, changed = sc, c, True
        print(f"round {rnd} after {k}: {cur} {best*100:+.2f}%", flush=True)
    if not changed: break
print("FINAL", json.dumps(cur), f"{best*100:+.2f}%")
for sd in (0, 1, 2):
    v, dg = rel(run(cur, sd), BASE[sd]); names = ["Apre", "Apost", "Bpre", "Bpost", "Cpre", "Cpost", "narma", "mackey", "regime"]
    print(f"seed set {sd}{' (held out)' if sd == 2 else ''}: " + "  ".join(f"{n} {x*100:+.0f}%" for n, x in zip(names, v)) + f"  digits {dg:+.2f}")
json.dump(cur, open("tuned_v5.json", "w"))
