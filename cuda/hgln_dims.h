// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_dims.h - the flat per-network layout of the HGLN kernel's state and scratch (HgDims). Plain C++ with no
// includes, so host code that only moves state (hgln_host.h, include/hgln_gpu.hpp, the mock drivers) can use it
// without the kernel or its real type. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#pragma once
#ifndef HG_HD
#define HG_HD
#endif
#define HG_MAXN 128        // largest supported input size d (prune scratch)
// Persistent state per network: H (3 x 64 hyperplanes x D, row (k*4+b) of layer l at (l*64 + k*4+b)*D, stored
// UNnormalised as the C++ library keeps them), Q (3 x 64 squared norms |H_b|^2, same row order), p (3 x 16 x D),
// W (dense U x 16 x N1 per layer), M and A (U x N1 per layer), m (3 x 16 x 16 competence map, only used with
// competence). |H_b| and 1/|H_b| are not stored: they are sqrt(Q) and 1/sqrt(Q), the same bits the library stores.
struct HgDims {            // flat per-network layout, shared by host and device
  int d, D, N1[3];
  long wofs[3], mofs[3], ioff[3];
  long Hs, Qs, ps, ms, Ws, Ms;                               // per-network sizes of the persistent arrays
  long o_u, o_inp, o_a, o_zn, o_pp, o_h, o_f, o_xm, o_pred, o_comp, o_r, o_es, o_cb, o_st, o_c2, o_sc, scs;   // scratch layout
};
HG_HD inline HgDims hg_dims(int d) {
  HgDims g; g.d = d; g.D = d + 1; g.N1[0] = d + 1; g.N1[1] = 17; g.N1[2] = 17;
  g.wofs[0] = 0; g.wofs[1] = 256L * g.N1[0]; g.wofs[2] = g.wofs[1] + 256L * 17; g.Ws = g.wofs[2] + 256L * 17;
  g.mofs[0] = 0; g.mofs[1] = 16L * g.N1[0]; g.mofs[2] = g.mofs[1] + 16L * 17; g.Ms = g.mofs[2] + 16L * 17;
  g.ioff[0] = 0; g.ioff[1] = g.N1[0]; g.ioff[2] = g.N1[0] + 17;
  g.Hs = 3L * 64 * g.D; g.Qs = 3L * 64; g.ps = 3L * 16 * g.D; g.ms = 3L * 256;
  long o = 0; g.o_u = o; o += g.D; g.o_inp = o; o += g.N1[0] + 34; g.o_a = o; o += 192; g.o_zn = o; o += 192;
  g.o_pp = o; o += 48; g.o_h = o; o += 768; g.o_f = o; o += 768; g.o_xm = o; o += g.Ms; g.o_pred = o; o += 48;
  g.o_comp = o; o += 48; g.o_r = o; o += 48; g.o_es = o; o += 48; g.o_cb = o; o += 192; g.o_st = o; o += 192;
  g.o_c2 = o; o += 48; g.o_sc = o; o += 8; g.scs = o;
  return g;
}
// Sample count after which the one-off prune has run: the first t >= max(warm, 1) with t % 64 == 0.
HG_HD inline long hg_prune_time(int warm) { long w = warm < 1 ? 1 : warm; return ((w + 63) / 64) * 64; }
