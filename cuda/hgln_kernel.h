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
//   * competence = 0 (4.3 default): the mixture is the plain average of the 3U units and there is no competence map;
//     competence = 1: the version 4.2 competence-weighted mixture (map m, responsibilities r).
//   * The per-sample scratch can live in dynamic shared memory (pass scratch = nullptr and launch with
//     hg_dims(d, units).scs * sizeof(hreal) bytes of dynamic shared memory) instead of global memory.
//   * units = Params::units (U = 8, 12 or 16 units per layer; the layout is hg_dims(d, U)); fanout = Params::fanout
//     (layer 3 reads layer 1). The gates of all layers run first, then the rule stages (see hgln_run below).
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

// One rule dot product f[k][c] = sum_j W[k][c][j] * xm[k][j] with the masked input xm = inp * M computed on the fly
// (the same products as storing xm first); xk != nullptr also stores xm (one task per unit does).
HG_DEV hreal hg_rule_dot(const hreal* Wrow, const hreal* in, const hreal* Mk, hreal* xk, int N1) {
  hreal sm = 0;
  for (int j = 0; j < N1; j++) { const hreal xv = in[j] * Mk[j]; if (xk) xk[j] = xv; sm += Wrow[j] * xv; }
  return sm;
}
// The 16-corner heatmap value of corner c from the unit's 4 gates (bit 0 first, as the library's tensor product).
HG_DEV hreal hg_corner(const hreal* ak, int c) {
  hreal v = 1;
  for (int b = 0; b < 4; b++) v *= ((c >> b) & 1) ? ak[b] : (1 - ak[b]);
  return v;
}
// Mixture of a unit's 16 rules by its heatmap (with competence: and its confidence 1 / (h . m + 1e-3)), clipped to
// [-2, 2]: the unit's prediction (Layer::rules_k).
HG_DEV hreal hg_unit_mix(const hreal* hk, const hreal* fk, const hreal* mk, hreal* compk) {
  hreal mix = 0;
  if (mk) {
    hreal cm = 0;
    for (int c = 0; c < 16; c++) { mix += hk[c] * fk[c]; cm += hk[c] * mk[c]; }
    *compk = 1 / (cm + (hreal)1e-3);
  } else {
    for (int c = 0; c < 16; c++) mix += hk[c] * fk[c];
  }
  return mix < -2 ? (hreal)-2 : (mix > 2 ? (hreal)2 : mix);
}

// X: T x d (shared). Y, P: nets x T. State arrays: nets x per-network size (hgln_dims.h, hg_dims(d, units)).
// tcount: samples already seen (in/out). scratch: nets x g.scs in global memory, or nullptr to use g.scs values of
// dynamic shared memory per block. competence: 0 = plain average (4.3), 1 = competence-weighted mixture (4.2); mall
// is only touched with 1. units: Params::units (must equal g.U; otherwise the kernel returns without doing anything).
// fanout: Params::fanout (layer 3 reads layer 1's outputs instead of layer 2's).
//
// Phases per sample (B = barrier; a phase's tasks are spread over the block's threads):
//   G   all 3 layers' gates (12U tasks: z_b, a_b, |p*u|^2) and layer 1's 16U rule dot products (inputs known)  B
//   M1  layer 1's mixture (computes its heatmap first) and the heatmaps of layers 2 and 3                      B
//   chain:  F2 B  M2 B  F3 B  M3 B        fanout: F2+F3 B  M2+M3 B      (F = 16U dot products, M = U mixtures)
//   U1  per-unit scalars; every thread forms the network mixture and error signals itself (no serial phase)   B
//   U2  p / hyperplane / rule / activity steps; thread 0 also prepares the next sample's u and input          B
//   (rare: R renormalisation B, the one-off prune B)
// = 8 barriers per sample for the chain, 6 with fanout (the earlier layer-by-layer kernel had 13).
// Every value is computed with the library's operation order (Net::infer / Layer::*_k / Net::learn).
HG_KERNEL void hgln_run(HgDims g, hreal* Hall, hreal* Qall, hreal* pall, hreal* Wall, hreal* Mall, hreal* Aall, hreal* mall,
                        hreal* scratch, const hreal* X, const hreal* Y, hreal* P, long* tcount, int T,
                        hreal glr0, hreal glr1, hreal tau, hreal glob, hreal mu, int warm, int competence, int units, int fanout) {
  HG_SMEM_DECL
  if (units != g.U) return;                                  // uniform: the whole block leaves before any barrier
  const int tid = HG_TID, nt = HG_NT, net = HG_BID, D = g.D, d = g.d, U = g.U, NU = 3 * U, U4 = 4 * U, U16 = 16 * U;
  const bool cmp = competence != 0, fan = fanout != 0;
  hreal* H = Hall + net * g.Hs; hreal* Q = Qall + net * g.Qs; hreal* p = pall + net * g.ps; hreal* W = Wall + net * g.Ws;
  hreal* M = Mall + net * g.Ms; hreal* A = Aall + net * g.Ms; hreal* m = cmp ? mall + net * g.ms : nullptr;
  hreal* S = scratch ? scratch + net * g.scs : HG_SMEM;
  hreal *ub = S + g.o_u, *inp = S + g.o_inp, *a = S + g.o_a, *zn = S + g.o_zn, *pq = S + g.o_pp, *h = S + g.o_h,
        *f = S + g.o_f, *xm = S + g.o_xm, *pred = S + g.o_pred, *comp = S + g.o_comp, *cr = S + g.o_cb, *st = S + g.o_st,
        *c2 = S + g.o_c2, *sc = S + g.o_sc;
  // sc[4 + l] = 1 if a squared norm of layer l left (1e-6, 1e6) this sample
  long t = tcount[net];
  const long tp = hg_prune_time(warm), RENORM = 4096;        // RENORM: hgln::Net::kRenormEvery
  const hreal BETA = 12;
  // side information u = [x;1]/|[x;1]| and the layer-1 input [x;1] of sample s, into buffer s & 1 (Net::infer)
#define HG_PREPARE(s_)                                                                                                \
  do {                                                                                                                \
    const hreal* x_ = X + (long)(s_) * d; hreal* u_ = ub + ((s_) & 1) * D; hreal* i_ = inp + ((s_) & 1) * g.N1[0]; \
    hreal nn = 1; for (int i = 0; i < d; i++) { u_[i] = x_[i]; nn += x_[i] * x_[i]; }                                 \
    u_[d] = 1; nn = HG_SQRT(nn); for (int i = 0; i <= d; i++) u_[i] /= nn;                                            \
    for (int j = 0; j < d; j++) i_[j] = x_[j];                                                                        \
    i_[d] = 1;                                                                                                        \
  } while (0)
  if (tid == 0) {
    inp[g.ioff[1] + U] = 1; inp[g.ioff[2] + U] = 1;          // bias entries of layers 2 and 3
    if (T > 0) HG_PREPARE(0);
  }
  HG_SYNC();
  for (int s = 0; s < T; s++) {
    const hreal* u = ub + (s & 1) * D;
    const hreal* in0 = inp + (s & 1) * g.N1[0];              // layer 1's input of this sample
    // ---- G: gates of all layers (they depend only on u) and layer 1's rule dot products
    HG_FOR(idx, 12 * U + U16) {
      if (idx < 12 * U) {                                    // z_b = (H_b . (p*u)) * (1/|H_b|); |p*u|^2 (once per unit)
        const int l = idx / U4, k = (idx % U4) / 4, b = idx % 4; hreal z = 0, s2 = 0;
        const hreal* Hb = H + ((long)l * U4 + k * 4 + b) * D; const hreal* pk = p + ((long)l * U + k) * D;
        if (b == 0) for (int i = 0; i < D; i++) { const hreal pu = pk[i] * u[i]; z += Hb[i] * pu; s2 += pu * pu; }
        else for (int i = 0; i < D; i++) z += Hb[i] * (pk[i] * u[i]);
        const hreal zz = z * ((hreal)1 / HG_SQRT(Q[idx]));
        zn[idx] = zz; a[idx] = hgln_gate<hreal>(zz);
        if (b == 0) pq[l * U + k] = s2;
      } else {                                               // layer 1: f[k][c], storing xm once per unit (c == 0)
        const int r = idx - 12 * U, k = r / 16, c = r % 16, N1 = g.N1[0];
        f[r] = hg_rule_dot(W + (long)r * N1, in0, M + (long)k * N1, c == 0 ? xm + (long)k * N1 : nullptr, N1);
      }
    }
    HG_SYNC();
    // ---- M1: layer 1's heatmaps + mixtures (one task per unit) and the heatmaps of layers 2 and 3
    if (tid == 0) { sc[4] = 0; sc[5] = 0; sc[6] = 0; }      // every thread has passed the previous sample's reads
    HG_FOR(idx, U + 2 * U16) {
      if (idx < U) {
        const int k = idx; hreal* hk = h + k * 16;
        for (int c = 0; c < 16; c++) hk[c] = hg_corner(a + k * 4, c);
        const hreal v = hg_unit_mix(hk, f + k * 16, cmp ? m + k * 16 : nullptr, comp + k);
        pred[k] = v;
        inp[g.ioff[1] + k] = v;                              // layer 2's input
        if (fan) inp[g.ioff[2] + k] = v;                     // fanout: layer 3 reads layer 1 too
      } else {
        const int r = idx - U + U16, gu = r / 16;            // r: index into h of layers 2 and 3
        h[r] = hg_corner(a + gu * 4, r % 16);
      }
    }
    HG_SYNC();
    // ---- rule stages of layers 2 and 3: one stage each for the chain, both together with fanout
    for (int l0 = 1; l0 < 3; l0 += fan ? 2 : 1) {
      const int nl = fan ? 2 : 1;                            // layers l0 .. l0 + nl - 1 in this stage
      HG_FOR(idx, nl * U16) {
        const int l = l0 + idx / U16, r = idx % U16, k = r / 16, c = r % 16, N1 = g.N1[l];
        f[l * U16 + r] = hg_rule_dot(W + g.wofs[l] + (long)r * N1, inp + g.ioff[l], M + g.mofs[l] + (long)k * N1,
                                     c == 0 ? xm + g.mofs[l] + (long)k * N1 : nullptr, N1);
      }
      HG_SYNC();
      HG_FOR(idx, nl * U) {
        const int l = l0 + idx / U, k = idx % U, gu = l * U + k;
        const hreal v = hg_unit_mix(h + gu * 16, f + gu * 16, cmp ? m + gu * 16 : nullptr, comp + gu);
        pred[gu] = v;
        if (l == 1 && !fan) inp[g.ioff[2] + k] = v;          // chain: layer 3's input
      }
      HG_SYNC();
    }
    // ---- U1: per-unit scalars (competence, gate steps, norms, rule step). Each thread with work forms the network
    // mixture and error signal itself from the 3U predictions (identical bits in every thread; Net::infer/learn).
    const bool need_A = t < tp;                              // activity only feeds the one-off prune
    if (tid < NU) {
      const hreal y = Y[(long)net * T + s], gl = glr1 + (glr0 - glr1) * HG_EXP(-(hreal)t / tau);
      hreal out = 0, cs = 0, sr2 = 0;
      if (cmp) {                                             // competence-weighted (4.2): r_i = comp_i / sum comp
        for (int i = 0; i < NU; i++) cs += comp[i];
        for (int i = 0; i < NU; i++) out += (comp[i] / cs) * pred[i];
        for (int i = 0; i < NU; i++) { const hreal ri = comp[i] / cs; sr2 += ri * ri; }
      } else {                                               // plain average (4.3): r_i = 1/(3U), r_i / sum r^2 = 1
        const hreal rr = (hreal)1 / NU;
        for (int i = 0; i < NU; i++) out += rr * pred[i];
      }
      if (tid == 0) P[(long)net * T + s] = out;
      const hreal eg = y - out;
      for (int gu = tid; gu < NU; gu += nt) {
        const int l = gu / U, k = gu % U, N1 = g.N1[l];
        hreal *hk = h + gu * 16, *fk = f + gu * 16, *ak = a + gu * 4;
        const hreal* xk = xm + g.mofs[l] + (long)k * N1;
        const hreal esk = cmp ? (1 - glob) * (y - pred[gu]) + glob * eg * (comp[gu] / cs) / sr2 : (1 - glob) * (y - pred[gu]) + glob * eg;
        if (cmp) {
          hreal* mk = m + gu * 16; const hreal e = y - pred[gu], e2 = e * e;
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
          const int j = gu * 4 + b;                          // = l * 4U + k * 4 + b
          const hreal q = Q[j], nq = HG_SQRT(q), rs = (hreal)1 / nq, s1 = cb * nq;
          cr[j] = cb * rs; st[j] = s1;
          const hreal q2 = q + (hreal)2 * s1 * (nq * zn[j]) + s1 * s1 * pq[gu];
          Q[j] = q2;
          far |= !(q2 > (hreal)1e-6 && q2 < (hreal)1e6);
        }
        if (far) sc[4 + l] = 1;                              // several threads may store the same 1
        hreal h2 = 0, x2 = 0;
        for (int c = 0; c < 16; c++) h2 += hk[c] * hk[c];
        for (int j = 0; j < N1; j++) x2 += xk[j] * xk[j];
        c2[gu] = (mu * esk) / ((hreal)1e-3 + h2 * x2);
      }
    }
    HG_SYNC();
    if (tid == 0 && s + 1 < T) HG_PREPARE(s + 1);            // the other buffer: nobody reads it before the barrier
    HG_FOR(idx, NU * D) {                                    // ---- U2a: fused p step (OLD H) + hyperplane step (OLD p)
      const int gu = idx / D, i = idx % D;
      hreal* Hk = H + (long)gu * 4 * D + i; hreal* pp = p + (long)gu * D + i;   // rows of unit gu = l * U + k
      const hreal* crk = cr + gu * 4; const hreal* stk = st + gu * 4;
      const hreal ui = u[i], pold = *pp, pu = pold * ui;
      const hreal h0 = Hk[0], h1 = Hk[D], h2 = Hk[2 * D], h3 = Hk[3 * D];
      const hreal dp = ((crk[0] * h0 * ui + crk[1] * h1 * ui) + crk[2] * h2 * ui) + crk[3] * h3 * ui;
      Hk[0] = h0 + stk[0] * pu; Hk[D] = h1 + stk[1] * pu; Hk[2 * D] = h2 + stk[2] * pu; Hk[3 * D] = h3 + stk[3] * pu;
      hreal v = pold + dp;
      *pp = v < 0 ? (hreal)0 : (v > 3 ? (hreal)3 : v);
    }
    HG_FOR(row, NU * 16) {                                   // ---- U2b: rule update (one row of W per task)
      const int gu = row / 16, c = row % 16, l = gu / U, k = gu % U, N1 = g.N1[l];
      const hreal sc2 = c2[gu] * h[row];
      hreal* Wc = W + g.wofs[l] + (long)(k * 16 + c) * N1; const hreal* xk = xm + g.mofs[l] + (long)k * N1;
      for (int j = 0; j < N1; j++) Wc[j] += sc2 * xk[j];
    }
    if (need_A) HG_FOR(gu, NU) {                             // ---- U2c: activity (until the prune)
      const int l = gu / U, k = gu % U, N1 = g.N1[l]; hreal* Ak = A + g.mofs[l] + (long)k * N1;
      const hreal* inl = l == 0 ? in0 : inp + g.ioff[l];
      for (int j = 0; j < N1; j++) Ak[j] = (hreal)0.98 * Ak[j] + (hreal)0.02 * HG_FABS(inl[j]);
    }
    HG_SYNC();
    // ---- R (rare): renormalise the layers whose norms drifted out of range (Layer::update_k), then, every RENORM
    // samples, every layer (Net::learn; after the prune there, which touches no hyperplane). Uniform condition.
    const bool far0 = sc[4] != 0, far1 = sc[5] != 0, far2 = sc[6] != 0, every = (t + 1) % RENORM == 0;
    if (far0 || far1 || far2 || every) {
      HG_FOR(row, 3 * U4) {                                  // row = l * 4U + k * 4 + b: H row at row * D
        const int l = row / U4;
        if (l == 0 ? far0 : (l == 1 ? far1 : far2)) hg_renorm_row(H + (long)row * D, Q + row, D);
        if (every) hg_renorm_row(H + (long)row * D, Q + row, D);
      }
      HG_SYNC();
    }
    t++;
    if (t == tp) {                                           // the one-off connection prune (later calls are no-ops)
      HG_FOR(gu, NU) { const int l = gu / U, k = gu % U; hg_prune_unit(g, l, k, W + g.wofs[l], M + g.mofs[l], A + g.mofs[l]); }
      HG_SYNC();
    }
  }
#undef HG_PREPARE
  if (tid == 0) tcount[net] = t;
}
