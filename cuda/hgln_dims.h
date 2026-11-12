// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_dims.h - the flat per-network layout of the HGLN kernel's state and scratch (HgDims). Plain C++ with no
// includes, so host code that only moves state (hgln_host.h, include/hgln_gpu.hpp, the mock drivers) can use it
// without the kernel or its real type. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#pragma once
#ifndef HG_HD
#define HG_HD
#endif
#define HG_MAXN 128        // largest supported input size d (prune scratch)
#define HG_MAXU 16         // largest number of units per layer (Params::units: 8, 12 or 16)
// U = units per layer (Params::units). Layer 1 takes the d inputs, layers 2 and 3 the U outputs of a previous layer,
// so N1 = d + 1, U + 1, U + 1 (inputs plus bias). Each unit has 4 hyperplanes and 16 corners (fixed).
// Persistent state per network: H (3 x 4U hyperplanes x D, row (k*4+b) of layer l at (l*4U + k*4+b)*D, stored
// UNnormalised as the C++ library keeps them), Q (3 x 4U squared norms |H_b|^2, same row order), p (3 x U x D),
// W (dense U x 16 x N1 per layer), M and A (U x N1 per layer), m (3 x U x 16 competence map, only used with
// competence). |H_b| and 1/|H_b| are not stored: they are sqrt(Q) and 1/sqrt(Q), the same bits the library stores.
struct HgDims {            // flat per-network layout, shared by host and device
  int d, D, U, N1[3];
  long wofs[3], mofs[3], ioff[3];
  long Hs, Qs, ps, ms, Ws, Ms;                               // per-network sizes of the persistent arrays
  long o_u, o_inp, o_a, o_zn, o_pp, o_h, o_f, o_xm, o_pred, o_comp, o_cb, o_st, o_c2, o_sc, o_kap, o_hold, o_cv, o_hs, o_sel, o_x2, scs;   // scratch layout
};
// Scratch (per network, in values of the real type): u and the layer-1 input x are double-buffered (u: 2 x D at
// o_u; inp: 2 x N1[0] for layer 1 at ioff[0] + (sample & 1) * N1[0], then layer 2's and layer 3's U + 1 inputs at
// ioff[1], ioff[2]) so the next sample's u is computed during the current sample's update; a, zn: 3 x 4U;
// pp, pred, comp, c2: 3U; h, f: 3 x 16U; xm: the masked inputs, like M; cb, st: 3U x 4; sc: 8 flags; version 5.0
// options (Params::latch, latch_credit): kap, hold (d a / d(12 z) and the latch's held part) and cv (the clipped latch
// credit each unit of layers 2 and 3 routes to its lower layer): 3U x 4 each; top-K corners (Params::corners < 16): hs, sel
// (the soft heatmap and the selection), 3 x 16U each; x2: each unit's |masked input|^2 (3U).
HG_HD inline HgDims hg_dims(int d, int U = 16) {
  HgDims g; g.d = d; g.D = d + 1; g.U = U; g.N1[0] = d + 1; g.N1[1] = U + 1; g.N1[2] = U + 1;
  const long R = 16L * U;                                    // rule rows per layer (U units x 16 corners)
  g.wofs[0] = 0; g.wofs[1] = R * g.N1[0]; g.wofs[2] = g.wofs[1] + R * g.N1[1]; g.Ws = g.wofs[2] + R * g.N1[2];
  g.mofs[0] = 0; g.mofs[1] = (long)U * g.N1[0]; g.mofs[2] = g.mofs[1] + (long)U * g.N1[1]; g.Ms = g.mofs[2] + (long)U * g.N1[2];
  g.ioff[0] = 0; g.ioff[1] = 2L * g.N1[0]; g.ioff[2] = g.ioff[1] + g.N1[1];
  g.Hs = 3L * 4 * U * g.D; g.Qs = 3L * 4 * U; g.ps = 3L * U * g.D; g.ms = 3L * R;
  long o = 0; g.o_u = o; o += 2L * g.D; g.o_inp = o; o += g.ioff[2] + g.N1[2]; g.o_a = o; o += 12L * U; g.o_zn = o; o += 12L * U;
  g.o_pp = o; o += 3L * U; g.o_h = o; o += 3 * R; g.o_f = o; o += 3 * R; g.o_xm = o; o += g.Ms; g.o_pred = o; o += 3L * U;
  g.o_comp = o; o += 3L * U; g.o_cb = o; o += 12L * U; g.o_st = o; o += 12L * U; g.o_c2 = o; o += 3L * U; g.o_sc = o; o += 8;
  g.o_kap = o; o += 12L * U; g.o_hold = o; o += 12L * U; g.o_cv = o; o += 12L * U; g.o_hs = o; o += 3 * R; g.o_sel = o; o += 3 * R; g.o_x2 = o; o += 3L * U;
  g.scs = o;
  return g;
}
// Device layout. On the device H, p, W, M and A are stored TRANSPOSED per network (and W, M, A per layer): element i
// of H row r at i * 12U + r, of p row (l U + k) at i * 3U + l U + k, element j of W row (k 16 + c) of layer l at
// wofs[l] + j * 16U + k 16 + c, of M / A (and the scratch xm) at mofs[l] + j * U + k. Each task still walks its own
// row in the library's order (the same bits), but adjacent threads now read adjacent addresses (coalesced). The
// host packs the row-major layout above and converts with hg_relayout (tmp: g.Hs + g.Ws values).
template <class T> HG_HD inline void hg_tr(T* a, T* tmp, long rows, long cols, bool to_dev) {
  for (long i = 0; i < rows * cols; i++) tmp[i] = a[i];
  for (long r = 0; r < rows; r++)
    for (long c = 0; c < cols; c++) { if (to_dev) a[c * rows + r] = tmp[r * cols + c]; else a[r * cols + c] = tmp[c * rows + r]; }
}
template <class T> HG_HD inline void hg_relayout(const HgDims& g, long nets, T* H, T* p, T* W, T* M, T* A, T* tmp, bool to_dev) {
  for (long n = 0; n < nets; n++) {
    hg_tr(H + n * g.Hs, tmp, 12L * g.U, g.D, to_dev); hg_tr(p + n * g.ps, tmp, 3L * g.U, g.D, to_dev);
    for (int l = 0; l < 3; l++) {
      hg_tr(W + n * g.Ws + g.wofs[l], tmp, 16L * g.U, g.N1[l], to_dev);
      hg_tr(M + n * g.Ms + g.mofs[l], tmp, (long)g.U, g.N1[l], to_dev); hg_tr(A + n * g.Ms + g.mofs[l], tmp, (long)g.U, g.N1[l], to_dev);
    }
  }
}
HG_HD inline bool hg_valid_units(int U) { return U == 8 || U == 12 || U == 16; }
// Sample count after which the one-off prune has run: the first t >= max(warm, 1) with t % 64 == 0.
HG_HD inline long hg_prune_time(int warm) { long w = warm < 1 ? 1 : warm; return ((w + 63) / 64) * 64; }
