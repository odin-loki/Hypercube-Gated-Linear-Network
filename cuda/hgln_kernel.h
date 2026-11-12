// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_kernel.h - HGLN (v4.2) batched kernel: one thread block per independent network; each block streams T samples
// (predict + learn per sample) entirely on the device. Networks share the input stream X and have their own targets Y.
// The same source compiles with nvcc (CUDA) and, through the macros below, as plain C++ for the CPU emulator
// (cuda_emu.cpp), which is how the logic was tested. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
//
// Changes from the first version (same results, see docs/OPTIMISATION.md):
//   * The per-sample scratch can live in dynamic shared memory (pass scratch = nullptr and launch with
//     hg_dims(d).scs * sizeof(hreal) bytes of dynamic shared memory) instead of global memory.
//   * pu = p * u is no longer stored: the p step and the hyperplane step are fused into one phase (one barrier fewer,
//     48 x D fewer scratch values).
//   * The activity trace A and the prune scan are skipped once the one-off prune has run (both are dead after it).
//   * The gate uses hgln_gate() and the renormalisation multiplies by 1/|H|, exactly as the C++ library does.
#pragma once
#ifndef HGLN_NO_STD_HEADERS     // the header-free PTX build (cuda/hgln_ptx.cu) cannot include host headers
#include <cmath>
#endif
#include "hgln_gate.h"
#ifndef HG_REAL
#define HG_REAL double
#endif
typedef HG_REAL hreal;
#ifndef HGLN_EMULATE
#define HG_KERNEL __global__
#define HG_DEV __device__ __forceinline__
#define HG_HD __host__ __device__
#define HG_TID (threadIdx.x)
#define HG_NT (blockDim.x)
#define HG_BID (blockIdx.x)
#define HG_SYNC() __syncthreads()
#define HG_SMEM_DECL extern __shared__ __align__(16) unsigned char hg_smem_raw[];
#define HG_SMEM (reinterpret_cast<hreal*>(hg_smem_raw))
#endif
#ifndef HG_HD
#define HG_HD
#endif
#ifndef HG_EXP              // device maths; hgln_ptx.cu (header-free clang build) supplies its own
#define HG_EXP(x) exp(x)
#define HG_SQRT(x) sqrt(x)
#define HG_FABS(x) fabs(x)
#define HG_RINT(x) nearbyint(x)
#endif
#define HG_MAXN 128        // largest supported input size d (prune scratch)
#define HG_FOR(i, N) for (int i = tid; i < (int)(N); i += nt)

#include "hgln_dims.h"        // HgDims, hg_dims(), hg_prune_time(): the flat layout, shared with the host code

// Prune one unit exactly as the CPU code does: keep round_half_even(0.6 n) inputs by mean|W| x activity.
HG_DEV void hg_prune_unit(const HgDims& g, int l, int k, hreal* W, hreal* M, const hreal* A) {
  int n = g.N1[l] - 1, N1 = g.N1[l], nt_keep = n <= 4 ? n : (int)HG_RINT(0.6 * (double)n);
  int act[HG_MAXN]; hreal keep[HG_MAXN]; int na = 0;
  for (int j = 0; j < n; j++) if (M[k * N1 + j] > 0) act[na++] = j;
  if (na <= nt_keep) return;
  for (int j = 0; j < n; j++) {
    hreal s = 0; for (int c = 0; c < 16; c++) s += HG_FABS(W[(k * 16 + c) * N1 + j]);
    keep[j] = (s / 16) * A[k * N1 + j];
  }
  for (int i = 1; i < na; i++) {                             // stable insertion sort, ascending
    int v = act[i], q = i - 1;
    while (q >= 0 && keep[act[q]] > keep[v]) { act[q + 1] = act[q]; q--; }
    act[q + 1] = v;
  }
  for (int i = 0; i < na - nt_keep; i++) M[k * N1 + act[i]] = 0;
}

// X: T x d (shared). Y, P: nets x T. State arrays: nets x per-network size. tcount: samples already seen (in/out).
// scratch: nets x g.scs in global memory, or nullptr to use g.scs values of dynamic shared memory per block.
HG_KERNEL void hgln_run(HgDims g, hreal* Hall, hreal* pall, hreal* Wall, hreal* Mall, hreal* Aall, hreal* mall,
                        hreal* scratch, const hreal* X, const hreal* Y, hreal* P, long* tcount, int T,
                        hreal glr0, hreal glr1, hreal tau, hreal glob, hreal mu, int warm) {
  HG_SMEM_DECL
  const int tid = HG_TID, nt = HG_NT, net = HG_BID, D = g.D, d = g.d;
  hreal* H = Hall + net * g.Hs; hreal* p = pall + net * g.ps; hreal* W = Wall + net * g.Ws;
  hreal* M = Mall + net * g.Ms; hreal* A = Aall + net * g.Ms; hreal* m = mall + net * g.ms;
  hreal* S = scratch ? scratch + net * g.scs : HG_SMEM;
  hreal *u = S + g.o_u, *inp = S + g.o_inp, *a = S + g.o_a, *h = S + g.o_h, *f = S + g.o_f,
        *xm = S + g.o_xm, *pred = S + g.o_pred, *comp = S + g.o_comp, *r = S + g.o_r, *es = S + g.o_es,
        *cb = S + g.o_cb, *c2 = S + g.o_c2, *sc = S + g.o_sc;     // sc: [0]=y [1]=gate rate
  long t = tcount[net];
  const long tp = hg_prune_time(warm);
  const hreal BETA = 12;
  if (tid == 0) { inp[g.ioff[1] + 16] = 1; inp[g.ioff[2] + 16] = 1; }   // bias entries of layers 2 and 3
  HG_SYNC();
  for (int s = 0; s < T; s++) {
    const hreal* x = X + (long)s * d;
    if (tid == 0) {                                          // side information u = [x;1]/|[x;1]|, layer-1 input
      hreal nn = 1; for (int i = 0; i < d; i++) { u[i] = x[i]; nn += x[i] * x[i]; }
      u[d] = 1; nn = HG_SQRT(nn); for (int i = 0; i <= d; i++) u[i] /= nn;
      for (int j = 0; j < d; j++) inp[j] = x[j];
      inp[d] = 1;
    }
    HG_SYNC();
    for (int l = 0; l < 3; l++) {                            // ---- forward, layer by layer
      const int N1 = g.N1[l];
      hreal *Hl = H + l * 64L * D, *pl = p + l * 16L * D, *Wl = W + g.wofs[l], *Ml = M + g.mofs[l], *ml = m + l * 256L;
      hreal *inl = inp + g.ioff[l], *al = a + l * 64, *hl = h + l * 256, *fl = f + l * 256, *xml = xm + g.mofs[l];
      HG_FOR(idx, 64) {
        int k = idx / 4, b = idx % 4; hreal z = 0;
        for (int i = 0; i < D; i++) z += Hl[(k * 4 + b) * D + i] * (pl[k * D + i] * u[i]);
        al[idx] = hgln_gate<hreal>(z);
      }
      HG_FOR(idx, 16 * N1) xml[idx] = inl[idx % N1] * Ml[idx];
      HG_SYNC();
      HG_FOR(idx, 256) {
        int k = idx / 16, c = idx % 16; hreal v = 1;
        for (int b = 0; b < 4; b++) v *= ((c >> b) & 1) ? al[k * 4 + b] : (1 - al[k * 4 + b]);
        hl[idx] = v;
        hreal sm = 0; for (int j = 0; j < N1; j++) sm += Wl[idx * N1 + j] * xml[k * N1 + j];
        fl[idx] = sm;
      }
      HG_SYNC();
      HG_FOR(k, 16) {
        hreal mix = 0, cm = 0;
        for (int c = 0; c < 16; c++) { mix += hl[k * 16 + c] * fl[k * 16 + c]; cm += hl[k * 16 + c] * ml[k * 16 + c]; }
        mix = mix < -2 ? (hreal)-2 : (mix > 2 ? (hreal)2 : mix);
        pred[l * 16 + k] = mix; comp[l * 16 + k] = 1 / (cm + (hreal)1e-3);
        if (l < 2) inp[g.ioff[l + 1] + k] = mix;             // next layer's input
      }
      HG_SYNC();
    }
    if (tid == 0) {                                          // ---- competitive mixture and error signals
      hreal cs = 0, out = 0, sr2 = 0;
      for (int i = 0; i < 48; i++) cs += comp[i];
      for (int i = 0; i < 48; i++) { r[i] = comp[i] / cs; out += r[i] * pred[i]; }
      P[(long)net * T + s] = out;
      hreal y = Y[(long)net * T + s], gl = glr1 + (glr0 - glr1) * HG_EXP(-(hreal)t / tau), eg = y - out;
      for (int i = 0; i < 48; i++) sr2 += r[i] * r[i];
      for (int i = 0; i < 48; i++) es[i] = (1 - glob) * (y - pred[i]) + glob * eg * r[i] / sr2;
      sc[0] = y; sc[1] = gl;
    }
    HG_SYNC();
    const hreal y = sc[0], gl = sc[1];
    const bool need_A = t < tp;                              // activity only feeds the one-off prune
    HG_FOR(gu, 48) {                                         // ---- U1: per-unit scalars (competence, gate steps, rule step)
      int l = gu / 16, k = gu % 16, N1 = g.N1[l];
      hreal *hk = h + l * 256 + k * 16, *fk = f + l * 256 + k * 16, *mk = m + l * 256L + k * 16, *ak = a + l * 64 + k * 4;
      hreal *xk = xm + g.mofs[l] + k * N1;
      hreal e = y - pred[gu], esk = es[gu];
      for (int c = 0; c < 16; c++) mk[c] += (hreal)0.02 * hk[c] * (e * e - mk[c]);
      hreal tot = 0, F1[4] = {0, 0, 0, 0};
      for (int c = 0; c < 16; c++) { hreal hf = hk[c] * fk[c]; tot += hf; for (int b = 0; b < 4; b++) if ((c >> b) & 1) F1[b] += hf; }
      for (int b = 0; b < 4; b++) {
        hreal F0 = tot - F1[b], gg = BETA * (F1[b] * (1 - ak[b]) - F0 * ak[b]), v = gl * esk * gg;
        cb[gu * 4 + b] = v < (hreal)-0.05 ? (hreal)-0.05 : (v > (hreal)0.05 ? (hreal)0.05 : v);
      }
      hreal h2 = 0, x2 = 0;
      for (int c = 0; c < 16; c++) h2 += hk[c] * hk[c];
      for (int j = 0; j < N1; j++) x2 += xk[j] * xk[j];
      c2[gu] = (mu * esk) / ((hreal)1e-3 + h2 * x2);
    }
    HG_SYNC();
    HG_FOR(idx, 48 * D) {                                    // ---- U2a: fused p step (OLD H) + hyperplane step (OLD p)
      int gu = idx / D, i = idx % D, l = gu / 16, k = gu % 16;
      hreal* Hl = H + l * 64L * D; hreal* pp = p + l * 16L * D + k * D + i;
      const hreal ui = u[i], pold = *pp, pu = pold * ui; hreal dp = 0;
      for (int b = 0; b < 4; b++) dp += cb[gu * 4 + b] * Hl[(k * 4 + b) * D + i] * ui;
      for (int b = 0; b < 4; b++) Hl[(k * 4 + b) * D + i] += cb[gu * 4 + b] * pu;
      hreal v = pold + dp;
      *pp = v < 0 ? (hreal)0 : (v > 3 ? (hreal)3 : v);
    }
    HG_FOR(row, 48 * 16) {                                   // ---- U2b: rule update (one row of W per task)
      int gu = row / 16, c = row % 16, l = gu / 16, k = gu % 16, N1 = g.N1[l];
      hreal sc2 = c2[gu] * h[l * 256 + k * 16 + c];
      hreal* Wc = W + g.wofs[l] + (long)(k * 16 + c) * N1; const hreal* xk = xm + g.mofs[l] + k * N1;
      for (int j = 0; j < N1; j++) Wc[j] += sc2 * xk[j];
    }
    if (need_A) HG_FOR(gu, 48) {                             // ---- U2c: activity (until the prune)
      int l = gu / 16, k = gu % 16, N1 = g.N1[l]; hreal* Ak = A + g.mofs[l] + k * N1; const hreal* inl = inp + g.ioff[l];
      for (int j = 0; j < N1; j++) Ak[j] = (hreal)0.98 * Ak[j] + (hreal)0.02 * HG_FABS(inl[j]);
    }
    HG_SYNC();
    HG_FOR(row, 48 * 4) {                                    // ---- U4: renormalise hyperplanes (multiply by 1/|H|)
      int gu = row / 4, b = row % 4, l = gu / 16, k = gu % 16; hreal* Hb = H + l * 64L * D + (k * 4 + b) * D; hreal nn = 0;
      for (int i = 0; i < D; i++) nn += Hb[i] * Hb[i];
      nn = (hreal)1 / HG_SQRT(nn); for (int i = 0; i < D; i++) Hb[i] *= nn;
    }
    HG_SYNC();
    t++;
    if (t == tp) {                                           // the one-off connection prune (later calls are no-ops)
      HG_FOR(gu, 48) { int l = gu / 16, k = gu % 16; hg_prune_unit(g, l, k, W + g.wofs[l], M + g.mofs[l], A + g.mofs[l]); }
      HG_SYNC();
    }
  }
  if (tid == 0) tcount[net] = t;
}
