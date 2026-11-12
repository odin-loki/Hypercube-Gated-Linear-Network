// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_kernel.h - HGLN batched kernel (the v5.0 options; 4.3 / 4.2 through the arguments): one thread block per independent network; each block streams T samples
// (predict + learn per sample) entirely on the device. Networks share the input stream X and have their own targets Y.
// The same source compiles with nvcc (CUDA) and, through the macros below, as plain C++ for the CPU emulator
// (cuda_emu.cpp), which is how the logic was tested; it has also run on an RTX 3090, bit-identical to the library. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
//
// Same algorithm and operation order as the C++ library (include/hgln.hpp), so float64 results are bit-identical:
//   * Norm tracking (4.3): hyperplanes H are stored unnormalised with their squared norms Q. The gate uses
//     z_b = (H_b . (p*u)) / |H_b|; the hyperplane step H_b += (cb |H_b|) p*u updates Q in closed form, so there is no
//     renormalisation pass per sample. A layer is renormalised only when one of its Q leaves (1e-6, 1e6), and every
//     layer every 4096 samples, exactly as Layer::update_k / Net::learn do.
//   * competence = 0 (default since 4.3): the mixture is the plain average of the 3U units and there is no competence map;
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
#define HG_WSYNC() __syncwarp()
#define HG_SMEM_DECL extern __shared__ __align__(16) unsigned char hg_smem_raw[];
#define HG_SMEM (reinterpret_cast<hreal*>(hg_smem_raw))
#endif
#ifndef HG_HD
#define HG_HD
#endif
#ifndef HG_WSYNC            // warp barrier (the warp kernel); the CPU emulators give each warp its own barrier
#define HG_WSYNC() HG_SYNC()
#endif
#ifndef HG_EXP              // device maths; hgln_ptx.cu (header-free clang build) supplies its own
#if defined(__CUDACC__)     // nvcc: the CUDA maths functions are overloaded for float
#define HG_EXP(x) exp(x)
#define HG_SQRT(x) sqrt(x)
#define HG_FABS(x) fabs(x)
#define HG_RINT(x) nearbyint(x)
#else                       // host (the CPU emulators): std:: keeps float as float, as the library and the GPU do
#define HG_EXP(x) std::exp(x)
#define HG_SQRT(x) std::sqrt(x)
#define HG_FABS(x) std::fabs(x)
#define HG_RINT(x) std::nearbyint(x)
#endif
#endif
#define HG_MAXN 128        // largest supported input size d (prune scratch)
#define HG_FOR(i, N) for (int i = tid; i < (int)(N); i += nt)
#ifndef HG_COLD             // rarely run code (kept out of line where the build allows)
#define HG_COLD HG_DEV
#endif
#ifndef HG_RESTRICT         // the kernel's arrays are separate allocations: no store can alias another array's loads
#define HG_RESTRICT __restrict__
#endif
#ifndef HG_UNROLL           // full unrolling keeps small fixed-size arrays in registers
#if defined(__clang__) || defined(__CUDACC__)
#define HG_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define HG_UNROLL _Pragma("GCC unroll 16")
#else
#define HG_UNROLL
#endif
#endif

#include "hgln_dims.h"        // HgDims, hg_dims(), hg_prune_time(): the flat layout, shared with the host code

// Prune one unit exactly as the CPU code does: keep round_half_even(0.6 n) inputs by mean|W| x activity.
HG_COLD void hg_prune_unit(const HgDims& g, int l, int k, hreal* W, hreal* M, const hreal* A) {
  int n = g.N1[l] - 1, N1 = g.N1[l], nt_keep = n <= 4 ? n : (int)HG_RINT(0.6 * (double)n);
  int act[HG_MAXN]; hreal keep[HG_MAXN]; int na = 0;
  const int U = g.U, U16 = 16 * g.U;                       // device layout: column j of unit k at j * U + k
  for (int j = 0; j < n; j++) if (M[j * U + k] > 0) act[na++] = j;
  if (na <= nt_keep) return;
  for (int j = 0; j < n; j++) {
    hreal s = 0; for (int c = 0; c < 16; c++) s += HG_FABS(W[(long)j * U16 + k * 16 + c]);
    keep[j] = (s / 16) * A[j * U + k];
  }
  for (int i = 1; i < na; i++) {                             // stable insertion sort, ascending
    int v = act[i], q = i - 1;
    while (q >= 0 && keep[act[q]] > keep[v]) { act[q + 1] = act[q]; q--; }
    act[q + 1] = v;
  }
  for (int i = 0; i < na - nt_keep; i++) M[act[i] * U + k] = 0;
  (void)N1;
}

// Scale hyperplane row Hb (D values) to unit length and recompute its squared norm from the result, as
// Layer::renormalise() does: H *= 1/sqrt(q), then q = sum_i H_i^2 (i ascending). Element i at Hb[i * ld].
HG_DEV void hg_renorm_row(hreal* Hb, hreal* q, int D, int ld) {
  const hreal rs = (hreal)1 / HG_SQRT(*q);
  for (int i = 0; i < D; i++) Hb[(long)i * ld] *= rs;
  hreal s2 = 0;
  for (int i = 0; i < D; i++) s2 += Hb[(long)i * ld] * Hb[(long)i * ld];
  *q = s2;
}

// One rule dot product f[k][c] = sum_j W[k][c][j] * xm[k][j] with the masked input xm = inp * M computed on the fly
// (the same products as storing xm first); xk != nullptr also stores xm (one task per unit does). Device layout:
// element j of the rule row at Wrow[j * ws], of the mask / masked input at Mk[j * ms] / xk[j * ms].
// Loops over a row load HG_CH elements ahead and then use them in order: the compiler cannot prove that a store (xk,
// W, A) does not alias the next loads, so a plain loop waits a full memory round trip per element. The arithmetic and
// its order are unchanged.
#ifndef HG_CH
#define HG_CH 8
#endif
#ifndef HG_U2               // U2a tasks per thread at once
#define HG_U2 4
#endif
HG_DEV hreal hg_rule_dot(const hreal* Wrow, int ws, const hreal* in, const hreal* Mk, hreal* xk, int ms, int N1, hreal* x2o) {
  hreal sm = 0, x2 = 0;                                      // xk: also |xm|^2 (j ascending, as U1 needs it) into *x2o
  for (int j0 = 0; j0 < N1; j0 += HG_CH) {
    const int n = N1 - j0 < HG_CH ? N1 - j0 : HG_CH;
    hreal wv[HG_CH], xv[HG_CH];
    HG_UNROLL for (int q = 0; q < HG_CH; q++) if (q < n) { const int j = j0 + q; wv[q] = Wrow[(long)j * ws]; xv[q] = in[j] * Mk[(long)j * ms]; }
    HG_UNROLL for (int q = 0; q < HG_CH; q++) if (q < n) { if (xk) { xk[(long)(j0 + q) * ms] = xv[q]; x2 += xv[q] * xv[q]; } sm += wv[q] * xv[q]; }
  }
  if (xk) *x2o = x2;
  return sm;
}
// Rule update of one row: Wc[j] += sc2 * xk[j] (element j at Wc[j * ws], xk[j * xs]).
HG_DEV void hg_rule_upd(hreal* Wc, int ws, const hreal* xk, int xs, hreal sc2, int N1) {
  for (int j0 = 0; j0 < N1; j0 += HG_CH) {
    const int n = N1 - j0 < HG_CH ? N1 - j0 : HG_CH;
    hreal wv[HG_CH], xv[HG_CH];
    HG_UNROLL for (int q = 0; q < HG_CH; q++) if (q < n) { wv[q] = Wc[(long)(j0 + q) * ws]; xv[q] = xk[(long)(j0 + q) * xs]; }
    HG_UNROLL for (int q = 0; q < HG_CH; q++) if (q < n) Wc[(long)(j0 + q) * ws] = wv[q] + sc2 * xv[q];
  }
}
// Activity trace of one unit: A[j] = 0.98 A[j] + 0.02 |in[j]| (element j at Ak[j * as]).
HG_DEV void hg_act_upd(hreal* Ak, int as, const hreal* in, int N1) {
  for (int j0 = 0; j0 < N1; j0 += HG_CH) {
    const int n = N1 - j0 < HG_CH ? N1 - j0 : HG_CH;
    hreal av[HG_CH], iv[HG_CH];
    HG_UNROLL for (int q = 0; q < HG_CH; q++) if (q < n) { av[q] = Ak[(long)(j0 + q) * as]; iv[q] = in[j0 + q]; }
    HG_UNROLL for (int q = 0; q < HG_CH; q++) if (q < n) Ak[(long)(j0 + q) * as] = (hreal)0.98 * av[q] + (hreal)0.02 * HG_FABS(iv[q]);
  }
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

// Cross-layer latch of one gate (Params::latch, Layer::gates_finish_k): set / reset sigmoids around +-theta, holding
// the lower layer's bit q0 in between. a = clip(st + (1 - st - rt) q0); kap = d a / d(12 z); hold = 1 - st - rt.
HG_DEV void hg_latch(hreal zz, hreal th, hreal q0, hreal* a, hreal* kap, hreal* hold) {
  const hreal BETA = 12;
  const hreal st = (hreal)1 / ((hreal)1 + hgln_exp<hreal>(-BETA * (zz - th))), rt = (hreal)1 / ((hreal)1 + hgln_exp<hreal>(-BETA * (-zz - th)));
  const hreal v = st + ((hreal)1 - st - rt) * q0;
  *a = v < (hreal)1e-4 ? (hreal)1e-4 : (v > (hreal)(1 - 1e-4) ? (hreal)(1 - 1e-4) : v);
  if (kap) *kap = st * ((hreal)1 - st) * ((hreal)1 - q0) + rt * ((hreal)1 - rt) * q0;
  if (hold) *hold = (hreal)1 - st - rt;
}
// Top-K corners of one unit (Params::corners = K < 16, Layer::top_corners): corner c is selected iff fewer than K
// corners have a larger soft value (ties to the lower corner); hk becomes the selected values renormalised to sum 1
// (sum over the selected corners, c ascending), 0 elsewhere; hs keeps the soft values and sel the selection (1 / 0).
HG_DEV void hg_top_corners(hreal* hk, hreal* hs, hreal* sel, int K) {
  hreal v[16], sm = 0; bool s[16];
  HG_UNROLL for (int c = 0; c < 16; c++) v[c] = hk[c];
  HG_UNROLL for (int c = 0; c < 16; c++) {
    int r = 0;
    HG_UNROLL for (int c2 = 0; c2 < 16; c2++) r += (int)(v[c2] > v[c]) | ((int)(v[c2] == v[c]) & (int)(c2 < c));
    s[c] = r < K;
  }
  HG_UNROLL for (int c = 0; c < 16; c++) if (s[c]) sm += v[c];
  HG_UNROLL for (int c = 0; c < 16; c++) { hs[c] = v[c]; sel[c] = s[c] ? (hreal)1 : (hreal)0; hk[c] = s[c] ? v[c] / sm : (hreal)0; }
}
// Gate step j (= l * 4U + k * 4 + b) applied to the tracked norm (Layer::update_k): cr = cb / |H_b|, st = cb |H_b|,
// |H_b + st pu|^2 = q + 2 st (|H_b| z_b) + st^2 |pu|^2. Returns true if the new squared norm left (1e-6, 1e6).
HG_DEV bool hg_gate_step(int j, hreal cb, hreal* Q, hreal* cr, hreal* st, const hreal* zn, hreal pqv) {
  const hreal q = Q[j], nq = HG_SQRT(q), rs = (hreal)1 / nq, s1 = cb * nq;
  cr[j] = cb * rs; st[j] = s1;
  const hreal q2 = q + (hreal)2 * s1 * (nq * zn[j]) + s1 * s1 * pqv;
  Q[j] = q2;
  return !(q2 > (hreal)1e-6 && q2 < (hreal)1e6);
}

// X: T x d (shared). Y, P: nets x T. State arrays: nets x per-network size (hgln_dims.h, hg_dims(d, units)).
// tcount: samples already seen (in/out). scratch: nets x g.scs in global memory, or nullptr to use g.scs values of
// dynamic shared memory per block. competence: 0 = plain average (4.3), 1 = competence-weighted mixture (4.2); mall
// is only touched with 1. units: Params::units (must equal g.U; otherwise the kernel returns without doing anything).
// fanout: Params::fanout (layer 3 reads layer 1's outputs instead of layer 2's). Version 5.0 options: latch =
// Params::latch (> 0: cross-layer latch on layers 2 and 3, holding the bits of the layer each reads from), credit =
// Params::latch_credit (> 0 with latch: the held part of the gate gradient, routed to the lower layer's gate step),
// gate_err = Params::gate_err (gate steps from 0 the mixed error, 1 the unit's own error, 2 the global error), corners =
// Params::corners (top-K corners per unit, 16 = all).
//
// Phases per sample (B = barrier; a phase's tasks are spread over the block's threads):
//   G   all 3 layers' gates (12U tasks: z_b, a_b, |p*u|^2) and layer 1's 16U rule dot products (inputs known)  B
//   M1  layer 1's mixture (computes its heatmap first) and the heatmaps of layers 2 and 3                      B
//   chain:  F2 B  M2 B  F3 B  M3 B        fanout: F2+F3 B  M2+M3 B      (F = 16U dot products, M = U mixtures)
//   U1  per-unit scalars; every thread forms the network mixture and error signals itself (no serial phase)   B
//       (latch credit: U1 stores the gate steps and the credit, B, then U1b adds the credit to the lower layers)
//   U2  p / hyperplane / rule / activity steps; thread 0 also prepares the next sample's u and input          B
//   (rare: R renormalisation B, the one-off prune B)
// = 8 barriers per sample for the chain, 6 with fanout (the earlier layer-by-layer kernel had 13); latch credit: + 1;
// latch: + 1 (M1a: one task per gate computes the latched gates, kap and hold; then M1b: one task per unit of every
// layer computes its heatmap and top-K).
// Every value is computed with the library's operation order (Net::infer / Layer::*_k / Net::learn).
//
// The per-sample code is one body run either by a whole thread block per network (hgln_run: tid / nt = the block's
// threads, barriers __syncthreads) or by one warp per network (hgln_wrun, CUDA: tid / nt = the warp's 32 lanes,
// barriers __syncwarp, several networks per block). Every task computes its value alone, in the library's order, so
// how the tasks are spread over threads never changes a bit.
#define HG_ARGS HgDims g, hreal* HG_RESTRICT Hall, hreal* HG_RESTRICT Qall, hreal* HG_RESTRICT pall, hreal* HG_RESTRICT Wall,        \
                hreal* HG_RESTRICT Mall, hreal* HG_RESTRICT Aall, hreal* HG_RESTRICT mall,                                 \
                hreal* HG_RESTRICT scratch, const hreal* HG_RESTRICT X, const hreal* HG_RESTRICT Y, hreal* HG_RESTRICT P,   \
                long* HG_RESTRICT tcount, int T,                                                                              \
                hreal glr0, hreal glr1, hreal tau, hreal glob, hreal mu, int warm, int competence, int units, int fanout, \
                hreal latch, hreal credit, int gate_err, int corners
#define HG_PASS g, Hall, Qall, pall, Wall, Mall, Aall, mall, scratch, X, Y, P, tcount, T, glr0, glr1, tau, glob, mu, warm, \
                competence, units, fanout, latch, credit, gate_err, corners
#define HG_BSYNC() do { if (WARP) HG_WSYNC(); else HG_SYNC(); } while (0)
template <bool WARP> HG_DEV void hgln_body(const int tid, const int nt, const int net, hreal* HG_RESTRICT smem, HG_ARGS) {
  if (units != g.U) return;                                  // uniform: the whole block / warp leaves before any barrier
  const int D = g.D, d = g.d, U = g.U, NU = 3 * U, U4 = 4 * U, U16 = 16 * U, R12 = 12 * U;
  const bool cmp = competence != 0, fan = fanout != 0, lat = latch > 0, cred = lat && credit > 0, topk = corners < 16;
  hreal* H = Hall + net * g.Hs; hreal* Q = Qall + net * g.Qs; hreal* p = pall + net * g.ps; hreal* W = Wall + net * g.Ws;
  hreal* M = Mall + net * g.Ms; hreal* A = Aall + net * g.Ms; hreal* m = cmp ? mall + net * g.ms : nullptr;
  hreal* S = scratch ? scratch + net * g.scs : smem;
  hreal *ub = S + g.o_u, *inp = S + g.o_inp, *a = S + g.o_a, *zn = S + g.o_zn, *pq = S + g.o_pp, *h = S + g.o_h,
        *f = S + g.o_f, *xm = S + g.o_xm, *pred = S + g.o_pred, *comp = S + g.o_comp, *cr = S + g.o_cb, *st = S + g.o_st,
        *c2 = S + g.o_c2, *sc = S + g.o_sc, *kap = S + g.o_kap, *hold = S + g.o_hold, *cv = S + g.o_cv, *hsa = S + g.o_hs, *sela = S + g.o_sel, *x2s = S + g.o_x2;
  // sc[4 + l] = 1 if a squared norm of layer l left (1e-6, 1e6) this sample
  long t = tcount[net];
  const long tp = hg_prune_time(warm), RENORM = 4096;        // RENORM: hgln::Net::kRenormEvery
  const hreal BETA = 12;
  // side information u = [x;1]/|[x;1]| and the layer-1 input [x;1] of sample s, into buffer s & 1 (Net::infer)
// Every thread computes the norm itself (the same loop, so the same bits) and the threads share the copies: no
// serial section. Run by all threads during U2 on the other buffer, which nobody reads before the next barrier.
#define HG_PREPARE(s_)                                                                                            \
  do {                                                                                                            \
    const hreal* x_ = X + (long)(s_) * d; hreal* u_ = ub + ((s_) & 1) * D; hreal* i_ = inp + ((s_) & 1) * g.N1[0];\
    hreal nn = 1; for (int i = 0; i < d; i++) nn += x_[i] * x_[i];                                                \
    nn = HG_SQRT(nn);                                                                                             \
    HG_FOR(i, D) { const hreal xv = i < d ? x_[i] : (hreal)1; u_[i] = xv / nn; i_[i] = xv; }                      \
  } while (0)
  if (tid == 0) { inp[g.ioff[1] + U] = 1; inp[g.ioff[2] + U] = 1; }   // bias entries of layers 2 and 3
  if (T > 0) HG_PREPARE(0);
  HG_BSYNC();
  for (int s = 0; s < T; s++) {
    const hreal* u = ub + (s & 1) * D;
    const hreal* in0 = inp + (s & 1) * g.N1[0];              // layer 1's input of this sample
    // ---- G: gates of all layers (they depend only on u) and layer 1's rule dot products
    HG_FOR(idx, 12 * U + U16) {
      if (idx < 12 * U) {                                    // z_b = (H_b . (p*u)) * (1/|H_b|); |p*u|^2 (once per unit)
        const int l = idx / U4, k = (idx % U4) / 4, b = idx % 4; hreal z = 0, s2 = 0;
        const hreal* Hb = H + idx; const hreal* pk = p + (l * U + k);   // row idx: element i at i * R12 (p: i * NU)
        for (int i0 = 0; i0 < D; i0 += HG_CH) {              // HG_CH loads in flight, then the products in order
          const int n = D - i0 < HG_CH ? D - i0 : HG_CH;
          hreal hv[HG_CH], pv[HG_CH], uv[HG_CH];
          HG_UNROLL for (int q = 0; q < HG_CH; q++) if (q < n) { const long i = i0 + q; hv[q] = Hb[i * R12]; pv[q] = pk[i * NU]; uv[q] = u[i]; }
          if (b == 0) { HG_UNROLL for (int q = 0; q < HG_CH; q++) if (q < n) { const hreal pu = pv[q] * uv[q]; z += hv[q] * pu; s2 += pu * pu; } }
          else HG_UNROLL for (int q = 0; q < HG_CH; q++) if (q < n) z += hv[q] * (pv[q] * uv[q]);
        }
        const hreal zz = z * ((hreal)1 / HG_SQRT(Q[idx]));
        zn[idx] = zz; a[idx] = hgln_gate<hreal>(zz);
        if (b == 0) pq[l * U + k] = s2;
      } else {                                               // layer 1: f[k][c], storing xm once per unit (c == 0)
        const int r = idx - 12 * U, k = r / 16, c = r % 16, N1 = g.N1[0];
        f[r] = hg_rule_dot(W + r, U16, in0, M + k, c == 0 ? xm + k : nullptr, U, N1, x2s + k);
      }
    }
    HG_BSYNC();
    // ---- M1: layer 1's heatmaps + mixtures (one task per unit) and the heatmaps of layers 2 and 3
    if (tid == 0) { sc[4] = 0; sc[5] = 0; sc[6] = 0; }      // every thread has passed the previous sample's reads
    if (lat) {                                               // M1a: the latched gates, kap / hold (one task per gate)
      HG_FOR(j, 3 * U4) {
        const int gu = j / 4, l = gu / U, k = gu % U, b = j % 4;
        if (l == 0) { kap[j] = a[j] * ((hreal)1 - a[j]); hold[j] = 0; }
        else {
          // the lower layer's (final) bit: layer 1's gates are never latched (no task writes them here); with the
          // chain, layer 3 holds layer 2's latched bit, recomputed here from layer 2's z (another task writes it)
          hreal q0 = a[k * 4 + b];
          if (!fan && l == 2) hg_latch(zn[(U + k) * 4 + b], latch, q0, &q0, nullptr, nullptr);
          hg_latch(zn[j], latch, q0, &a[j], &kap[j], &hold[j]);
        }
      }
      HG_BSYNC();
    }
    if (lat || topk) HG_FOR(gu, NU) {                      // M1b, per unit: heatmap, top-K, layer 1's mixture
      const int l = gu / U, k = gu % U; const hreal* ak = a + gu * 4;
      hreal* hk = h + gu * 16;
      HG_UNROLL for (int c = 0; c < 16; c++) hk[c] = hg_corner(ak, c);
      if (topk) hg_top_corners(hk, hsa + gu * 16, sela + gu * 16, corners);
      if (l == 0) {
        const hreal v = hg_unit_mix(hk, f + k * 16, cmp ? m + k * 16 : nullptr, comp + k);
        pred[k] = v;
        inp[g.ioff[1] + k] = v;
        if (fan) inp[g.ioff[2] + k] = v;
      }
    }
    else HG_FOR(idx, U + 2 * U16) {
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
    HG_BSYNC();
    // ---- rule stages of layers 2 and 3: one stage each for the chain, both together with fanout
    for (int l0 = 1; l0 < 3; l0 += fan ? 2 : 1) {
      const int nl = fan ? 2 : 1;                            // layers l0 .. l0 + nl - 1 in this stage
      HG_FOR(idx, nl * U16) {
        const int l = l0 + idx / U16, r = idx % U16, k = r / 16, c = r % 16, N1 = g.N1[l];
        f[l * U16 + r] = hg_rule_dot(W + g.wofs[l] + r, U16, inp + g.ioff[l], M + g.mofs[l] + k,
                                     c == 0 ? xm + g.mofs[l] + k : nullptr, U, N1, x2s + l * U + k);
      }
      HG_BSYNC();
      HG_FOR(idx, nl * U) {
        const int l = l0 + idx / U, k = idx % U, gu = l * U + k;
        const hreal v = hg_unit_mix(h + gu * 16, f + gu * 16, cmp ? m + gu * 16 : nullptr, comp + gu);
        pred[gu] = v;
        if (l == 1 && !fan) inp[g.ioff[2] + k] = v;          // chain: layer 3's input
      }
      HG_BSYNC();
    }
    // ---- U1: per-unit scalars (competence, gate steps, norms, rule step). Each thread with work forms the network
    // mixture and error signal itself from the 3U predictions (identical bits in every thread; Net::infer/learn).
    const bool need_A = t < tp;                              // activity only feeds the one-off prune
    if (tid < NU) {
      const hreal y = Y[(long)net * T + s], gl = glr1 + (glr0 - glr1) * hgln_exp<hreal>(-(hreal)t / tau);
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
        const int l = gu / U, k = gu % U;
        hreal *hk = h + gu * 16, *fk = f + gu * 16, *ak = a + gu * 4;
        const hreal esk = cmp ? (1 - glob) * (y - pred[gu]) + glob * eg * (comp[gu] / cs) / sr2 : (1 - glob) * (y - pred[gu]) + glob * eg;
        if (cmp) {
          hreal* mk = m + gu * 16; const hreal e = y - pred[gu], e2 = e * e;
          for (int c = 0; c < 16; c++) mk[c] += (hreal)0.02 * hk[c] * (e2 - mk[c]);
        }
        hreal tot = 0, F1[4] = {0, 0, 0, 0};
        const hreal *hsk = hsa + gu * 16, *slk = sela + gu * 16;   // top-K: gradient from the soft values of the selected corners
        for (int c = 0; c < 16; c++) { hreal hf = topk ? (hsk[c] * fk[c]) * slk[c] : hk[c] * fk[c]; tot += hf; for (int b = 0; b < 4; b++) if ((c >> b) & 1) F1[b] += hf; }
        // error driving the gate steps (Params::gate_err): mixed, the unit's own, or the global error
        const hreal gev = gate_err == 2 ? eg : (gate_err == 1 ? y - pred[gu] : esk);
        const bool latched = lat && l > 0;
        bool far = false;
        for (int b = 0; b < 4; b++) {
          const int j = gu * 4 + b;                          // = l * 4U + k * 4 + b
          const hreal F0 = tot - F1[b];
          hreal v;
          if (latched) {                                     // g = 12 kap (F1/a - F0/(1-a)); credit: the held part
            const hreal dm = F1[b] / ak[b] - F0 / ((hreal)1 - ak[b]);
            v = gl * gev * (BETA * kap[j] * dm);
            if (cred) {
              const int lk = ((fan ? 0 : l - 1) * U + k) * 4 + b;   // the same unit's bit in the lower layer
              const hreal c1 = gl * gev * (credit * BETA * kap[lk] * dm * hold[j]);
              cv[j] = c1 < (hreal)-0.05 ? (hreal)-0.05 : (c1 > (hreal)0.05 ? (hreal)0.05 : c1);
            }
          } else {
            const hreal gg = BETA * (F1[b] * (1 - ak[b]) - F0 * ak[b]);
            v = gl * gev * gg;
          }
          const hreal cb = v < (hreal)-0.05 ? (hreal)-0.05 : (v > (hreal)0.05 ? (hreal)0.05 : v);
          // OLD norms: p step uses H_b/|H_b| (cr = cb/|H_b|), hyperplane step H_b += (cb |H_b|) p*u (st); then
          // |H_b + st pu|^2 = q + 2 st (H_b . pu) + st^2 |pu|^2 with H_b . pu = |H_b| z_b
          if (cred) cr[j] = cb;                              // the credit from the upper layers is added in U1b
          else far |= hg_gate_step(j, cb, Q, cr, st, zn, pq[gu]);
        }
        if (far) sc[4 + l] = 1;                              // several threads may store the same 1
        hreal h2 = 0, x2 = 0;
        for (int c = 0; c < 16; c++) h2 += hk[c] * hk[c];
        x2 = x2s[gu];                                        // |xm|^2, summed by the unit's c == 0 rule task
        c2[gu] = (mu * esk) / ((hreal)1e-3 + h2 * x2);
      }
    }
    HG_BSYNC();
    if (cred) {                                              // ---- U1b: gate step + the latch credit from the upper layers
      HG_FOR(j, 3 * U4) {                                    // Net::learn updates layers 3, 2, 1: acc = (0 + c3) + c2
        const int gu = j / 4, l = gu / U, k = gu % U, b = j % 4;
        hreal acc = 0;
        for (int up = 2; up > l; up--) if ((fan ? 0 : up - 1) == l) acc += cv[(up * U + k) * 4 + b];
        if (hg_gate_step(j, cr[j] + acc, Q, cr, st, zn, pq[gu])) sc[4 + l] = 1;
      }
      HG_BSYNC();
    }
    if (s + 1 < T) HG_PREPARE(s + 1);                        // the other buffer: nobody reads it before the barrier
    // ---- U2a: fused p step (OLD H) + hyperplane step (OLD p); each thread takes HG_U2 tasks at once (loads first)
    for (int base = tid; base < NU * D; base += HG_U2 * nt) {
      hreal hv[HG_U2][4], pv[HG_U2], uv[HG_U2];
      HG_UNROLL for (int q = 0; q < HG_U2; q++) {
        const int idx = base + q * nt;
        if (idx < NU * D) {
          const int gu = idx % NU, i = idx / NU;             // adjacent tasks: adjacent units (adjacent addresses)
          const hreal* Hk = H + (long)i * R12 + gu * 4;      // element i of unit gu's 4 rows / p
          hv[q][0] = Hk[0]; hv[q][1] = Hk[1]; hv[q][2] = Hk[2]; hv[q][3] = Hk[3]; pv[q] = p[(long)i * NU + gu]; uv[q] = u[i];
        }
      }
      HG_UNROLL for (int q = 0; q < HG_U2; q++) {
        const int idx = base + q * nt;
        if (idx < NU * D) {
          const int gu = idx % NU, i = idx / NU;
          hreal* Hk = H + (long)i * R12 + gu * 4;
          const hreal* crk = cr + gu * 4; const hreal* stk = st + gu * 4;
          const hreal ui = uv[q], pold = pv[q], pu = pold * ui;
          const hreal h0 = hv[q][0], h1 = hv[q][1], h2 = hv[q][2], h3 = hv[q][3];
          const hreal dp = ((crk[0] * h0 * ui + crk[1] * h1 * ui) + crk[2] * h2 * ui) + crk[3] * h3 * ui;
          Hk[0] = h0 + stk[0] * pu; Hk[1] = h1 + stk[1] * pu; Hk[2] = h2 + stk[2] * pu; Hk[3] = h3 + stk[3] * pu;
          const hreal v = pold + dp;
          p[(long)i * NU + gu] = v < 0 ? (hreal)0 : (v > 3 ? (hreal)3 : v);
        }
      }
    }
    HG_FOR(row, NU * 16) {                                   // ---- U2b: rule update (one row of W per task)
      const int gu = row / 16, c = row % 16, l = gu / U, k = gu % U, N1 = g.N1[l];
      const hreal sc2 = c2[gu] * h[row];
      hreal* Wc = W + g.wofs[l] + k * 16 + c; const hreal* xk = xm + g.mofs[l] + k;
      hg_rule_upd(Wc, U16, xk, U, sc2, N1);
    }
    if (need_A) HG_FOR(gu, NU) {                             // ---- U2c: activity (until the prune)
      const int l = gu / U, k = gu % U, N1 = g.N1[l]; hreal* Ak = A + g.mofs[l] + k;
      const hreal* inl = l == 0 ? in0 : inp + g.ioff[l];
      hg_act_upd(Ak, U, inl, N1);
    }
    HG_BSYNC();
    // ---- R (rare): renormalise the layers whose norms drifted out of range (Layer::update_k), then, every RENORM
    // samples, every layer (Net::learn; after the prune there, which touches no hyperplane). Uniform condition.
    const bool far0 = sc[4] != 0, far1 = sc[5] != 0, far2 = sc[6] != 0, every = (t + 1) % RENORM == 0;
    if (far0 || far1 || far2 || every) {
      HG_FOR(row, 3 * U4) {                                  // row = l * 4U + k * 4 + b: H row at row * D
        const int l = row / U4;
        if (l == 0 ? far0 : (l == 1 ? far1 : far2)) hg_renorm_row(H + row, Q + row, D, R12);
        if (every) hg_renorm_row(H + row, Q + row, D, R12);
      }
      HG_BSYNC();
    }
    t++;
    if (t == tp) {                                           // the one-off connection prune (later calls are no-ops)
      HG_FOR(gu, NU) { const int l = gu / U, k = gu % U; hg_prune_unit(g, l, k, W + g.wofs[l], M + g.mofs[l], A + g.mofs[l]); }
      HG_BSYNC();
    }
  }
#undef HG_PREPARE
  if (tid == 0) tcount[net] = t;
}

// X: T x d (shared). Y, P: nets x T. State arrays: nets x per-network size (hgln_dims.h, hg_dims(d, units)). One
// thread block per network (any block size); scratch: global (nets x g.scs) or, with scratch = nullptr, g.scs values
// of dynamic shared memory per block.
HG_KERNEL void hgln_run(HG_ARGS) {
  HG_SMEM_DECL
  hgln_body<false>(HG_TID, HG_NT, HG_BID, HG_SMEM, HG_PASS);
}
#ifndef HGLN_NO_WARP_KERNEL
// One warp per network: blockDim = 32 W runs networks [W blockIdx, W blockIdx + W) (nets in total); with scratch =
// nullptr each warp's scratch is its own g.scs values of dynamic shared memory (W g.scs per block).
HG_KERNEL void hgln_wrun(HG_ARGS, int nets) {
  HG_SMEM_DECL
  const int t = (int)HG_TID, w = t >> 5, net = (int)HG_BID * ((int)HG_NT >> 5) + w;
  if (net >= nets) return;                                   // whole warps only: no block barrier follows
  hgln_body<true>(t & 31, 32, net, HG_SMEM + (long)w * g.scs, HG_PASS);
}
#endif
