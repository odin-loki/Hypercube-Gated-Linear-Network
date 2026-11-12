# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare variants in results.jsonl with the baseline: late windows (pre-drift 1536-2048, post-drift 3584-4096),
mean NMSE change and seed wins (lower is better); digits after 3 passes, mean change and shuffle wins; cost ratios."""
import json, sys
R = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")]
B = next(r for r in R if r["name"] == "baseline")
def row(r):
    out = [f"{r['name']:<28}"]
    for t in "ABC":
        for w in (1, 3):
            m = r["syn"][t]["mean"][w]; b = B["syn"][t]["mean"][w]
            wins = sum(a[w] < c[w] for a, c in zip(r["syn"][t]["seeds"], B["syn"][t]["seeds"]))
            out.append(f"{t}{'pre' if w == 1 else 'post'} {m:.3f}({m - b:+.3f},{wins})")
    d = r["digits"]["mean"][2]; db = B["digits"]["mean"][2]
    dw = sum(a[2] > c[2] for a, c in zip(r["digits"]["shuffles"], B["digits"]["shuffles"]))
    out.append(f"dig {d:.2f}({d - db:+.2f},{dw})")
    for k in ("d8", "d64"):
        out.append(f"{k} macs x{r['cost'][k]['macs'] / B['cost'][k]['macs']:.2f} ram x{r['cost'][k]['state'] / B['cost'][k]['state']:.2f}")
    return "  ".join(out)
for r in R: print(row(r))
