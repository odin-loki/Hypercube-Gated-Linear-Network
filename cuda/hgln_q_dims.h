// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_q_dims.h - flat per-network layout and constants of the integer kernel (cuda/hgln_q_kernel.h). Plain C++ with
// no includes beyond hgln_dims.h, shared by the host (hgln_q_host.h, src/gpu) and the device.
// Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#pragma once
#include "hgln_dims.h"
#define HQ_MAXN 129        // largest layer input count incl. the bias (d <= 128)
#if defined(__NVPTX__) || defined(__AMDGPU__)
static_assert(sizeof(long) == 8, "the kernels' HgDims / HgQDims and tcount use 64-bit long (LP64), as the host does");
#endif
// Constants of a network (include/hgln_qspec.hpp formats; hgln::QCfg and Params), passed by value.
struct HgQCfg {
  int AB, WB, FA, FF, FH, FP, MA, MW, ONE;   // qspec::Fmt
  int K, latch, credit, gate_err, fanout, warm, units;
  int th14;                                  // latch dead band, Q14 (round_param(theta, FA) << (16 - AB))
  long long cscale, G, Mq;                   // latch credit scale, glob and mu, Q16
  long long g0, g1; double tau;              // gate rate schedule: glr0, glr1 at Q24 and tau (qspec::gl_q24)
};
// Persistent state per network, int32 (the stored int16 values), dense like the research reference QNet:
// H (layer l, unit k, bit b, input i at ((l U + k) 4 + b) D + i), p ((l U + k) D + i), W (layer l at woff[l]: unit k,
// corner c, input j at (k 16 + c) N1[l] + j), M and A (layer l at moff[l]: k N1[l] + j; M = 1 for an active input).
// Scratch per network (scs int64 values): the float32 hyperplane rows Hr and row scales (in the 12 U D and 12 U int64
// slots at o_Hr, o_rc), then int32 arrays at o_* (offsets in int32 units from
// the end of the int64 part): s (D), x (the layer inputs incl. bias, at xoff[l]), z (12 U), a, kap, hold (12 U),
// hs, h, f, sel (3 x 16 U), o (3 U), cbr, cr (12 U), sc (3 x 16 U), pn (3 U x D).
struct HgQDims {
  int d, D, U, N1[3];
  long woff[3], moff[3], xoff[3];
  long Hs, ps, Ws, Ms;
  long o_Hr, o_rc, n64;
  long o_s, o_x, o_z, o_a, o_kap, o_hold, o_hs, o_h, o_f, o_sel, o_o, o_cbr, o_cr, o_sc, o_pn, n32;
  long scs;                                  // int64 values per network
};
HG_HD inline HgQDims hq_dims(int d, int U) {
  HgQDims g; g.d = d; g.D = d + 1; g.U = U; g.N1[0] = d + 1; g.N1[1] = U + 1; g.N1[2] = U + 1;
  long w = 0, m = 0, x = 0;
  for (int l = 0; l < 3; l++) { g.woff[l] = w; w += 16L * U * g.N1[l]; g.moff[l] = m; m += (long)U * g.N1[l]; g.xoff[l] = x; x += g.N1[l]; }
  g.Ws = w; g.Ms = m; g.Hs = 3L * U * 4 * g.D; g.ps = 3L * U * g.D;
  long o = 0; g.o_Hr = o; o += 12L * U * g.D; g.o_rc = o; o += 12L * U; g.n64 = o;
  long q = 0; const long R = 16L * U;
  g.o_s = q; q += g.D; g.o_x = q; q += x; g.o_z = q; q += 12L * U; g.o_a = q; q += 12L * U; g.o_kap = q; q += 12L * U; g.o_hold = q; q += 12L * U;
  g.o_hs = q; q += 3 * R; g.o_h = q; q += 3 * R; g.o_f = q; q += 3 * R; g.o_sel = q; q += 3 * R; g.o_o = q; q += 3L * U;
  g.o_cbr = q; q += 12L * U; g.o_cr = q; q += 12L * U; g.o_sc = q; q += 3 * R; g.o_pn = q; q += 3L * U * g.D; g.n32 = q;
  g.scs = g.n64 + (q + 1) / 2;
  return g;
}
