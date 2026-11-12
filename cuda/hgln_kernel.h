// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_kernel.h - HGLN (v4.3) batched kernel: one thread block per independent network; each block streams T samples
// (predict + learn per sample) entirely on the device. Networks share the input stream X and have their own targets Y.
// The same source compiles with nvcc (CUDA) and, through the macros below, as plain C++ for the CPU emulator
// (cuda_emu.cpp), which is how the logic was tested. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
//
// Same algorithm and operation order as the C++ library (include/hgln.hpp), so float64 results are bit-identical:
//   * Norm tracking (4.3): hyperplanes H are stored unnormalised with their squared norms Q. The gate uses
//     z_b = (H_b . (p*u)) / |H_b|; the hyperplane step H_b += (cb |H_b|) p*u updates Q in closed form, so there is no
//     renormalisation pass per sample. A layer is renormalised only when one of its Q leaves (1e-6, 1e6), and every
//     layer every 4096 samples, exactly as Layer::update_k / Net::learn do.
//   * competence = 0 (4.3 default): the mixture is the plain average of the 48 units and there is no competence map;
//     competence = 1: the version 4.2 competence-weighted mixture (map m, responsibilities r).
//   * The per-sample scratch can live in dynamic shared memory (pass scratch = nullptr and launch with
//     hg_dims(d).scs * sizeof(hreal) bytes of dynamic shared memory) instead of global memory.
//   * pu = p * u is not stored: the p step and the hyperplane step are fused into one phase.
//   * The activity trace A and the prune scan are skipped once the one-off prune has run (both are dead after it).
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

// Scale hyperplane row Hb (D values) to unit length and recompute its squared norm from the result, as
// Layer::renormalise() does: H *= 1/sqrt(q), then q = sum_i H_i^2 (i ascending).
HG_DEV void hg_renorm_row(hreal* Hb, hreal* q, int D) {
  const hreal rs = (hreal)1 / HG_SQRT(*q);
  for (int i = 0; i < D; i++) Hb[i] *= rs;
  hreal s2 = 0;
  for (int i = 0; i < D; i++) s2 += Hb[i] * Hb[i];
  *q = s2;
}

// X: T x d (shared). Y, P: nets x T. State arrays: nets x per-network size (hgln_dims.h). tcount: samples already
// seen (in/out). scratch: nets x g.scs in global memory, or nullptr to use g.scs values of dynamic shared memory per
// block. competence: 0 = plain average (4.3), 1 = competence-weighted mixture (4.2); mall is only touched with 1.
HG_KERNEL void hgln_run(HgDims g, hreal* Hall, hreal* Qall, hreal* pall, hreal* Wall, hreal* Mall, hreal* Aall, hreal* mall,
                        hreal* scratch, const hreal* X, const hreal* Y, hreal* P, long* tcount, int T,
                        hreal glr0, hreal glr1, hreal tau, hreal glob, hreal mu, int warm, int competence) {
  HG_SMEM_DECL
  const int tid = HG_TID, nt = HG_NT, net = HG_BID, D = g.D, d = g.d;
  const bool cmp = competence != 0;
  hreal* H = Hall + net * g.Hs; hreal* Q = Qall + net * g.Qs; hreal* p = pall + net * g.ps; hreal* W = Wall + net * g.Ws;
  hreal* M = Mall + net * g.Ms; hreal* A = Aall + net * g.Ms; hreal* m = cmp ? mall + net * g.ms : nullptr;
  hreal* S = scratch ? scratch + net * g.scs : HG_SMEM;
  hreal *u = S + g.o_u, *inp = S + g.o_inp, *a = S + g.o_a, *zn = S + g.o_zn, *pq = S + g.o_pp, *h = S + g.o_h,
        *f = S + g.o_f, *xm = S + g.o_xm, *pred = S + g.o_pred, *comp = S + g.o_comp, *r = S + g.o_r, *es = S + g.o_es,
        *cr = S + g.o_cb, *st = S + g.o_st, *c2 = S + g.o_c2, *sc = S + g.o_sc;
  // sc: [0] = y, [1] = gate rate, [4 + l] = 1 if a squared norm of layer l left (1e-6, 1e6) this sample
  long t = tcount[net];
  const long tp = hg_prune_time(warm), RENORM = 4096;        // RENORM: hgln::Net::kRenormEvery
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
      hreal *Hl = H + l * 64L * D, *Ql = Q + l * 64, *pl = p + l * 16L * D, *Wl = W + g.wofs[l], *Ml = M + g.mofs[l];
      hreal *inl = inp + g.ioff[l], *al = a + l * 64, *znl = zn + l * 64, *hl = h + l * 256, *fl = f + l * 256, *xml = xm + g.mofs[l];
      HG_FOR(idx, 64) {                                      // z_b = (H_b . (p*u)) * (1/|H_b|); |p*u|^2 (once per unit)
        int k = idx / 4, b = idx % 4; hreal z = 0, s2 = 0;
        const hreal* Hb = Hl + (k * 4 + b) * D; const hreal* pk = pl + k * D;
        if (b == 0) for (int i = 0; i < D; i++) { const hreal pu = pk[i] * u[i]; z += Hb[i] * pu; s2 += pu * pu; }
        else for (int i = 0; i < D; i++) z += Hb[i] * (pk[i] * u[i]);
        const hreal zz = z * ((hreal)1 / HG_SQRT(Ql[k * 4 + b]));
        znl[idx] = zz; al[idx] = hgln_gate<hreal>(zz);
        if (b == 0) pq[l * 16 + k] = s2;
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
        hreal mix = 0;
        if (cmp) {
          const hreal* ml = m + l * 256L; hreal cm = 0;
          for (int c = 0; c < 16; c++) { mix += hl[k * 16 + c] * fl[k * 16 + c]; cm += hl[k * 16 + c] * ml[k * 16 + c]; }
          comp[l * 16 + k] = 1 / (cm + (hreal)1e-3);
        } else {
          for (int c = 0; c < 16; c++) mix += hl[k * 16 + c] * fl[k * 16 + c];
        }
        mix = mix < -2 ? (hreal)-2 : (mix > 2 ? (hreal)2 : mix);
        pred[l * 16 + k] = mix;
        if (l < 2) inp[g.ioff[l + 1] + k] = mix;             // next layer's input
      }
      HG_SYNC();
    }
    if (tid == 0) {                                          // ---- mixture and error signals
      hreal out = 0;
      hreal y = Y[(long)net * T + s], gl = glr1 + (glr0 - glr1) * HG_EXP(-(hreal)t / tau);
      if (cmp) {                                             // competence-weighted (4.2)
        hreal cs = 0, sr2 = 0;
        for (int i = 0; i < 48; i++) cs += comp[i];
        for (int i = 0; i < 48; i++) { r[i] = comp[i] / cs; out += r[i] * pred[i]; }
        const hreal eg = y - out;
        for (int i = 0; i < 48; i++) sr2 += r[i] * r[i];
        for (int i = 0; i < 48; i++) es[i] = (1 - glob) * (y - pred[i]) + glob * eg * r[i] / sr2;
      } else {                                               // plain average (4.3): r_i = 1/48, r_i / sum r^2 = 1
        const hreal rr = (hreal)1 / 48;
        for (int i = 0; i < 48; i++) out += rr * pred[i];
        const hreal eg = y - out;
        for (int i = 0; i < 48; i++) es[i] = (1 - glob) * (y - pred[i]) + glob * eg;
      }
      P[(long)net * T + s] = out;
      sc[0] = y; sc[1] = gl; sc[4] = 0; sc[5] = 0; sc[6] = 0;
    }
    HG_SYNC();
    const hreal y = sc[0], gl = sc[1];
    const bool need_A = t < tp;                              // activity only feeds the one-off prune
    HG_FOR(gu, 48) {                                         // ---- U1: per-unit scalars (competence, gate steps, norms, rule step)
      int l = gu / 16, k = gu % 16, N1 = g.N1[l];
      hreal *hk = h + l * 256 + k * 16, *fk = f + l * 256 + k * 16, *ak = a + l * 64 + k * 4;
      hreal *xk = xm + g.mofs[l] + k * N1;
      hreal esk = es[gu];
      if (cmp) {
        hreal* mk = m + l * 256L + k * 16; const hreal e = y - pred[gu], e2 = e * e;
        for (int c = 0; c < 16; c++) mk[c] += (hreal)0.02 * hk[c] * (e2 - mk[c]);
      }
      hreal tot = 0, F1[4] = {0, 0, 0, 0};
      for (int c = 0; c < 16; c++) { hreal hf = hk[c] * fk[c]; tot += hf; for (int b = 0; b < 4; b++) if ((c >> b) & 1) F1[b] += hf; }
      bool far = false;
      for (int b = 0; b < 4; b++) {
        hreal F0 = tot - F1[b], gg = BETA * (F1[b] * (1 - ak[b]) - F0 * ak[b]), v = gl * esk * gg;
        const hreal cb = v < (hreal)-0.05 ? (hreal)-0.05 : (v > (hreal)0.05 ? (hreal)0.05 : v);
        // OLD norms: p step uses H_b/|H_b| (cr = cb/|H_b|), hyperplane step H_b += (cb |H_b|) p*u (st); then
        // |H_b + st pu|^2 = q + 2 st (H_b . pu) + st^2 |pu|^2 with H_b . pu = |H_b| z_b
        const int j = l * 64 + k * 4 + b;
        const hreal q = Q[j], nq = HG_SQRT(q), rs = (hreal)1 / nq, s1 = cb * nq;
        cr[gu * 4 + b] = cb * rs; st[gu * 4 + b] = s1;
        const hreal q2 = q + (hreal)2 * s1 * (nq * zn[j]) + s1 * s1 * pq[gu];
        Q[j] = q2;
        far |= !(q2 > (hreal)1e-6 && q2 < (hreal)1e6);
      }
      if (far) sc[4 + l] = 1;                                // several threads may store the same 1
      hreal h2 = 0, x2 = 0;
      for (int c = 0; c < 16; c++) h2 += hk[c] * hk[c];
      for (int j = 0; j < N1; j++) x2 += xk[j] * xk[j];
      c2[gu] = (mu * esk) / ((hreal)1e-3 + h2 * x2);
    }
    HG_SYNC();
    HG_FOR(idx, 48 * D) {                                    // ---- U2a: fused p step (OLD H) + hyperplane step (OLD p)
      int gu = idx / D, i = idx % D, l = gu / 16, k = gu % 16;
      hreal* Hk = H + l * 64L * D + (k * 4) * D + i; hreal* pp = p + l * 16L * D + k * D + i;
      const hreal* crk = cr + gu * 4; const hreal* stk = st + gu * 4;
      const hreal ui = u[i], pold = *pp, pu = pold * ui;
      const hreal h0 = Hk[0], h1 = Hk[D], h2 = Hk[2 * D], h3 = Hk[3 * D];
      const hreal dp = ((crk[0] * h0 * ui + crk[1] * h1 * ui) + crk[2] * h2 * ui) + crk[3] * h3 * ui;
      Hk[0] = h0 + stk[0] * pu; Hk[D] = h1 + stk[1] * pu; Hk[2 * D] = h2 + stk[2] * pu; Hk[3 * D] = h3 + stk[3] * pu;
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
    // ---- R (rare): renormalise the layers whose norms drifted out of range (Layer::update_k), then, every RENORM
    // samples, every layer (Net::learn; after the prune there, which touches no hyperplane). Uniform condition.
    const bool far0 = sc[4] != 0, far1 = sc[5] != 0, far2 = sc[6] != 0, every = (t + 1) % RENORM == 0;
    if (far0 || far1 || far2 || every) {
      HG_FOR(row, 192) {                                     // row = l * 64 + k * 4 + b: H row at row * D
        const int l = row / 64;
        if (l == 0 ? far0 : (l == 1 ? far1 : far2)) hg_renorm_row(H + (long)row * D, Q + row, D);
        if (every) hg_renorm_row(H + (long)row * D, Q + row, D);
      }
      HG_SYNC();
    }
    t++;
    if (t == tp) {                                           // the one-off connection prune (later calls are no-ops)
      HG_FOR(gu, 48) { int l = gu / 16, k = gu % 16; hg_prune_unit(g, l, k, W + g.wofs[l], M + g.mofs[l], A + g.mofs[l]); }
      HG_SYNC();
    }
  }
  if (tid == 0) tcount[net] = t;
}
