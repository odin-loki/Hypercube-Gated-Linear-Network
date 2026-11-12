# SPDX-License-Identifier: AGPL-3.0-or-later
"""Baselines for the text8 benchmark (research/nlp/text8_hgln.cpp): character n-gram models of order 1..6 with
interpolated absolute discounting, P_n(c|h) = max(N(h,c) - D, 0) / N(h) + D T(h) / N(h) P_{n-1}(c|h') (T: distinct
followers; unseen h: P_{n-1}; order 0: uniform over the 27 symbols), the discount D tuned on validation per order.
Trained on characters [0, 90M), validation [90M, 95M), test [95M, 100M). Prints one JSON line per order.
usage: python3 text8_ngram.py text8 [max_order=6]"""
import json, sys, time
import numpy as np
path = sys.argv[1]; maxo = int(sys.argv[2]) if len(sys.argv) > 2 else 6
raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
s = np.where(raw == 32, 0, raw.astype(np.int64) - 96)
assert s.min() >= 0 and s.max() <= 26 and len(s) == 100_000_000
C = 27; TR, V0, T0, NV = 90_000_000, 90_000_000, 95_000_000, 5_000_000
def ctx_codes(lo, hi, k):        # context code of the k symbols before each position in [lo, hi) (-1 where t < k)
    t = np.arange(lo, hi); code = np.zeros(hi - lo, dtype=np.int64); ok = t >= k
    for i in range(1, k + 1): code = code * C + np.where(t - i >= 0, s[np.maximum(t - i, 0)], 0)
    return np.where(ok, code, -1)
tables = {}
for k in range(1, maxo):         # context length k = order - 1
    t0 = time.time(); h = ctx_codes(k, TR, k); y = s[k:TR]
    hc = np.unique(h * C + y, return_counts=True); hh = np.unique(h, return_counts=True)
    th = np.unique(hc[0] // C, return_counts=True)   # distinct followers per context
    tables[k] = (hc, hh, th)
    print(f"  order {k + 1}: {len(hh[0])} contexts, {len(hc[0])} (context, symbol) pairs, {time.time() - t0:.0f} s", file=sys.stderr)
uni = np.bincount(s[:TR], minlength=C).astype(np.float64)
def look(keys, sorted_keys, vals):
    i = np.searchsorted(sorted_keys, keys); i = np.minimum(i, len(sorted_keys) - 1)
    hit = sorted_keys[i] == keys; return np.where(hit, vals[i], 0)
def probs(lo, n, order, D):      # P(s[t] | context) for t in [lo, lo + n) with interpolated absolute discounting
    y = s[lo:lo + n]
    p = np.full(n, 1.0 / C)
    p = (np.maximum(uni[y] - D[0], 0) + D[0] * C * p) / uni.sum() if order >= 1 else p   # order 1: discounted unigram over uniform
    for k in range(1, order):
        hc, hh, th = tables[k]; h = ctx_codes(lo, lo + n, k)
        nhc = look(h * C + y, hc[0], hc[1]).astype(np.float64); nh = look(h, hh[0], hh[1]).astype(np.float64); tt = look(h, th[0], th[1]).astype(np.float64)
        seen = nh > 0
        p = np.where(seen, (np.maximum(nhc - D[k], 0) + D[k] * tt * p) / np.maximum(nh, 1), p)
    return p
for order in range(1, maxo + 1):
    D = [0.5] * order
    for k in range(order):       # tune this order's discount on validation, the lower ones already fixed
        best = None
        for dk in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
            D[k] = dk; b = -np.mean(np.log2(probs(V0, NV, order, D)))
            if best is None or b < best[0]: best = (b, dk)
        D[k] = best[1]
    vb = -np.mean(np.log2(probs(V0, NV, order, D))); tb = -np.mean(np.log2(probs(T0, NV, order, D)))
    print(json.dumps({"model": f"{order}-gram, interpolated absolute discounting", "discounts": D, "valid_bpc": round(float(vb), 4), "test_bpc": round(float(tb), 4)}), flush=True)
