# SPDX-License-Identifier: AGPL-3.0-or-later
"""Markdown table of results.jsonl against the baseline (lower NMSE is better; seed wins out of 10, shuffle wins out of 5)."""
import json, sys
R = {}
for l in open(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl"): r = json.loads(l); R[r["name"]] = r
B = R["baseline"]
order = sys.argv[2].split(",") if len(sys.argv) > 2 else list(R)
print("| Variant | A pre | A post | B pre | B post | C pre | C post | Digits 3 passes | Ops d=8 | Ops d=64 | State d=64 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for n in order:
    r = R[n]; c = [n]
    for t in "ABC":
        for w in (1, 3):
            m = r["syn"][t]["mean"][w]; wins = sum(a[w] < b[w] for a, b in zip(r["syn"][t]["seeds"], B["syn"][t]["seeds"]))
            c.append(f"{m:.3f}" if n == "baseline" else f"{m:.3f} ({wins}/10)")
    d = r["digits"]["mean"][2]; dw = sum(a[2] > b[2] for a, b in zip(r["digits"]["shuffles"], B["digits"]["shuffles"]))
    c.append(f"{d:.2f}" if n == "baseline" else f"{d:.2f} ({dw}/5)")
    for k, f in (("d8", "macs"), ("d64", "macs"), ("d64", "state")):
        c.append(f"{r['cost'][k][f] / B['cost'][k][f]:.2f}x")
    print("| " + " | ".join(c) + " |")
