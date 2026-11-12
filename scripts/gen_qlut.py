# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the integer sigmoid table of the HGLN v5.0 integer datapath (include/hgln_qspec.hpp, kSigLut).
Entry i (0..1024) = round_half_even(16384 * clip(1 / (1 + exp(-12 z)), 1e-4, 1 - 1e-4)), z = (i - 512) / 512.
Computed with 60-digit decimal arithmetic, so no entry depends on a floating-point library. Author: Odin Loch."""
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
getcontext().prec = 60
lo, hi = Decimal("1e-4"), 1 - Decimal("1e-4")
out = []
for i in range(1025):
    z = Decimal(i - 512) / 512
    s = 1 / (1 + (-12 * z).exp())
    s = min(max(s, lo), hi)
    v = (16384 * s)
    q = v.quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
    assert abs(v - q) != Decimal("0.5")
    out.append(int(q))
assert all(out[i] + out[1024 - i] == 16384 for i in range(1025))
lines = []
for i in range(0, 1025, 24):
    lines.append("    " + ",".join(str(v) for v in out[i:i + 24]) + ",")
print("\n".join(lines))
