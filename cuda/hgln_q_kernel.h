// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_q_kernel.h - the version 5.0 integer network (hgln::NetQ, include/hgln_q.hpp) as a batched GPU kernel: one
// thread block per independent network, each block streams T samples (predict + learn) on the device. It runs the
// integer specification of include/hgln_qspec.hpp exactly (written like its research reference, QNet in
// research/alg/hgln_var.hpp, in a dense layout), so every result is bit-identical to the CPU library on any device:
// the forward pass and the per-unit control scalars are integer arithmetic, and the per-element learning steps are the
// spec's mixed float32 steps (IEEE single precision, the same operations in the same order as the CPU, no fused
// multiply-adds), so every device gives the CPU's bits. The other floating-point operations are the conversions of
// the caller's inputs and targets, the double estimates inside the exact integer division and square root, and the
// returned prediction.
// Included after hgln_kernel.h (it uses its HG_* macros and hreal). Copyright (C) Odin Loch. GNU AGPL v3.0 or later.
//
// Differences from the float kernel: the spec's two tables (the 1025-entry Q14 sigmoid table and e^-n at Q30 for the
// gate rate schedule) are kernel arguments in global memory; the scratch is always in global memory (its int64
// hyperplane rows are large).
#pragma once
#include "hgln_q_dims.h"

// ---- the spec's integer helpers (include/hgln_qspec.hpp), restated for the device
HG_DEV long long hq_rne(long long v, int k) {                  // v / 2^k, nearest, ties to even; k <= 0: v * 2^-k
  if (k <= 0) return (long long)((unsigned long long)v << (-k));
  return (v + (1LL << (k - 1)) - 1 + ((v >> k) & 1)) >> k;
}
HG_DEV float hq_rnf(float v) { return (v + 12582912.0f) - 12582912.0f; }   // qspec::rnf: |v| < 2^22 to nearest even
HG_DEV long long hq_clamp(long long v, long long lo, long long hi) { return v < lo ? lo : (v > hi ? hi : v); }
HG_DEV long long hq_sat(long long v, long long m) { return hq_clamp(v, -m, m); }
HG_DEV long long hq_sat32(long long v) { return hq_clamp(v, -2147483647LL, 2147483647LL); }
HG_DEV long long hq_div_rne(long long n, long long d) {        // n / d to nearest, ties to even, d > 0 (exact)
  long long q, r;
  const double qa = (double)n / (double)d;
  if (qa > -1.0e15 && qa < 1.0e15) {
    q = (long long)qa; r = n - q * d;
    while (r < 0) { q -= 1; r += d; }
    while (r >= d) { q += 1; r -= d; }
  } else {
    q = n / d; r = n % d;
    if (r < 0) { q -= 1; r += d; }
  }
  if (2 * r > d || (2 * r == d && (q & 1))) q += 1;
  return q;
}
HG_DEV long long hq_divd(long long n, long long d) {   // qspec::divd: the double quotient to nearest even, +-(2^31 - 1)
  const double q = HG_RINT((double)n / (double)d);
  return q >= 2147483647.0 ? 2147483647LL : (q <= -2147483647.0 ? -2147483647LL : (long long)q);
}
HG_DEV unsigned long long hq_isqrt(unsigned long long n) {     // floor(sqrt(n)), exact, n < 2^62
  unsigned long long r = (unsigned long long)HG_SQRT((double)n);
  while (r * r > n) r--;
  while ((r + 1) * (r + 1) <= n) r++;
  return r;
}
HG_DEV int hq_quantise(double x, int frac, int m) {           // clamp, then round to nearest even; NaN -> 0
  if (!(x == x)) return 0;
  const double v = x * (double)(1LL << frac);
  if (v >= (double)m) return m;
  if (v <= -(double)m) return -m;
  return (int)HG_RINT(v);
}
HG_DEV int hq_sig14(const short* lut, int v) {                 // the spec's sig14: table lookup with linear interpolation
  const int vc = (v < -16384 ? -16384 : (v > 16383 ? 16383 : v)) + 16384, i = vc >> 5, r = vc & 31;
  return lut[i] + (((lut[i + 1] - lut[i]) * r + 16) >> 5);
}
// gate a (Q(FA)), kap and hold (Q14) of one bit: plain (q0 < 0) or the cross-layer latch holding the lower bit q0
HG_DEV void hq_gate(const short* lut, const HgQCfg& C, int z14, int q0, int* a, int* kap, int* hold) {
  const int sh14 = 16 - C.AB;
  if (q0 >= 0) {
    const int st = hq_sig14(lut, z14 - C.th14), rt = hq_sig14(lut, -z14 - C.th14), q14 = q0 * (1 << sh14);
    const long long a14 = hq_clamp(st + hq_rne((long long)(16384 - st - rt) * q14, 14), 2, 16382);
    if (kap) *kap = (int)(hq_rne(hq_rne((long long)st * (16384 - st), 14) * (16384 - q14), 14) + hq_rne(hq_rne((long long)rt * (16384 - rt), 14) * q14, 14));
    if (hold) *hold = 16384 - st - rt;
    *a = (int)hq_clamp(hq_rne(a14, sh14), 1, C.ONE - 1);
  } else {
    *a = (int)hq_clamp(hq_rne(hq_sig14(lut, z14), sh14), 1, C.ONE - 1);
    const long long aq = (long long)*a << sh14;
    if (kap) *kap = (int)hq_rne(aq * (16384 - aq), 14);
    if (hold) *hold = 0;
  }
}
// gl(t) at Q24 (qspec::gl_q24): exp(-t / tau) from e^-n (Q30, table) times a 14-term series for the fraction
HG_DEV long long hq_gl(const HgQCfg& C, const long long* expneg, long t) {
  const double xd = (double)t * 4294967296.0 / C.tau;
  long long E = 0;
  if (!(xd >= 22.0 * 4294967296.0)) {
    const unsigned long long X = (unsigned long long)HG_RINT(xd), n = X >> 32;
    if (n < 22) {
      const long long fr = (long long)((X & 0xFFFFFFFFull) >> 2);
      long long term = 1LL << 30, sum = term;
      for (int k = 1; k <= 14; k++) { term = ((term * fr) >> 30) / k; sum += (k & 1) ? -term : term; }
      E = (expneg[n] * sum) >> 30;
    }
  }
  return C.g1 + hq_rne((C.g0 - C.g1) * E, 30);
}
// 12 kap (F1/a - F0/(1-a)) at Q24 (the latched gate gradient)
HG_DEV long long hq_dz(const HgQCfg& C, long long kap, long long num, long long den) {
  const int sh = C.FF - 10; long long nu = 12 * kap * num, de = den;
  if (sh >= 0) de <<= sh; else nu *= 1LL << -sh;
  return hq_sat32(hq_divd(nu, de));
}

// Hall, pall: 3 x U x 4 x D / 3 x U x D; Wall, Mall, Aall: per layer U x 16 x N1 / U x N1 (hgln_q_dims.h), all int32
// (the stored int16 values). scratch: nets x g.scs int64. lut: the 1025-entry sigmoid table (qspec::kSigLut);
// expneg: e^-n at Q30, n = 0..21 (qspec::kExpNeg). X: T x d (shared); Y, P: nets x T. tcount: samples seen (in/out).
//
// Phases per sample (B = barrier): Q (thread 0: quantised input, side information) B; G (all gate dot products,
// layer 1's rule outputs) B; M1 (per unit: gates incl. the latch, kap / hold, heatmap, top-K, layer 1's mixture) B;
// rule stages of layers 2 and 3 (fan-out: together) F B M B (chain: twice); U1 (per unit: errors, gate steps, latch
// credit, LMS scalars) B; U1b (gate step + credit from the upper layers) B; U2a (hyperplane rows, p, W, activity) B;
// U2b (renormalised H, p) B; the one-off prune B.
HG_KERNEL void hglnq_run(HgQDims g, HgQCfg C, int* Hall, int* pall, int* Wall, int* Mall, int* Aall, long long* scratch,
                         const short* lut, const long long* expneg, const hreal* X, const hreal* Y, hreal* P, long* tcount, int T) {
  if (C.units != g.U) return;
  const int tid = HG_TID, nt = HG_NT, net = HG_BID, D = g.D, d = g.d, U = g.U, NU = 3 * U, U4 = 4 * U, U16 = 16 * U;
  const bool fan = C.fanout != 0, lat = C.latch != 0, cred = lat && C.credit != 0;
  const int sh14 = 16 - C.AB;
  const long long CB = 53687091;                              // 0.05 at Q30
  const float KP = 1.0f / (float)(1LL << (29 + C.AB)), KH = 1.0f / (float)(1LL << (27 + C.AB)), KW = 1.0f / (float)(1LL << (26 + C.AB - C.WB));
  int *H = Hall + net * g.Hs, *p = pall + net * g.ps, *W = Wall + net * g.Ws, *M = Mall + net * g.Ms, *A = Aall + net * g.Ms;
  long long* S64 = scratch + net * g.scs;
  float* Hr = reinterpret_cast<float*>(S64 + g.o_Hr); float* gsc = reinterpret_cast<float*>(S64 + g.o_rc);   // float32 rows, row scales
  int* S32 = reinterpret_cast<int*>(S64 + g.n64);
  int *s = S32 + g.o_s, *x = S32 + g.o_x, *z14 = S32 + g.o_z, *a = S32 + g.o_a, *kap = S32 + g.o_kap, *hold = S32 + g.o_hold,
      *hs = S32 + g.o_hs, *h = S32 + g.o_h, *f = S32 + g.o_f, *sel = S32 + g.o_sel, *o = S32 + g.o_o, *cbr = S32 + g.o_cbr,
      *cr = S32 + g.o_cr, *sc = S32 + g.o_sc, *pn = S32 + g.o_pn;
  long t = tcount[net];
  const long tp = hg_prune_time(C.warm);
  const long long G = C.G, Mq = C.Mq;
  for (int smp = 0; smp < T; smp++) {
    // ---- Q: quantised input (layer 1's input) and the side information s = [x, 1] / |[x, 1]|
    if (tid == 0) {
      const hreal* xs = X + (long)smp * d;
      long long nn = (long long)C.ONE * C.ONE;
      for (int j = 0; j < d; j++) { const int v = hq_quantise((double)xs[j], C.FA, C.MA); x[g.xoff[0] + j] = v; nn += (long long)v * v; }
      x[g.xoff[0] + d] = C.ONE; x[g.xoff[1] + U] = C.ONE; x[g.xoff[2] + U] = C.ONE;   // bias inputs
      const long long gq = (long long)hq_isqrt((unsigned long long)nn << 6), rcp = (1LL << (34 + C.FA)) / gq;
      for (int j = 0; j < d; j++) s[j] = (int)hq_rne((long long)x[g.xoff[0] + j] * rcp, 31);
      s[d] = (int)hq_rne((long long)C.ONE * rcp, 31);
    }
    HG_SYNC();
    // ---- G: the gate pre-activations of all layers (they depend only on s) and layer 1's rule outputs
    HG_FOR(idx, 12 * U + U16) {
      if (idx < 12 * U) {                                    // idx = (l * U + k) * 4 + b
        const int gu = idx / 4;
        const int* Hb = H + (long)idx * D; const int* pk = p + (long)gu * D;
        unsigned z = 0;
        for (int i = 0; i < D; i++) z += (unsigned)(Hb[i] * (int)hq_sat(hq_rne((long long)pk[i] * s[i], C.FP), C.MA));
        z14[idx] = (int)hq_sat(hq_rne((int)z, C.FH), C.MA) * (1 << sh14);
      } else {                                               // layer 1: f[k][c]
        const int r = idx - 12 * U, k = r / 16, N1 = g.N1[0];
        const int* Wr = W + g.woff[0] + (long)r * N1; const int* Mk = M + g.moff[0] + (long)k * N1; const int* xl = x + g.xoff[0];
        unsigned acc = 0;
        for (int j = 0; j < N1; j++) if (Mk[j]) acc += (unsigned)(Wr[j] * xl[j]);
        f[r] = (int)hq_sat(hq_rne((int)acc, C.WB - 2), C.MA);
      }
    }
    HG_SYNC();
    // ---- M1: per unit of every layer: its gates (the latch holds the lower layer's bit: layer 1, or with the chain
    // layer 2 for layer 3, recomputed here from z), kap / hold, heatmap, top-K corners; layer 1's mixture
    HG_FOR(gu, NU) {
      const int l = gu / U, k = gu % U;
      int ak[4];
      for (int b = 0; b < 4; b++) {
        const int j = gu * 4 + b;
        if (!lat || l == 0) { hq_gate(lut, C, z14[j], -1, &ak[b], &kap[j], &hold[j]); continue; }
        int q0; hq_gate(lut, C, z14[k * 4 + b], -1, &q0, nullptr, nullptr);               // layer 1's bit
        if (!fan && l == 2) hq_gate(lut, C, z14[(U + k) * 4 + b], q0, &q0, nullptr, nullptr);   // layer 2's latched bit
        hq_gate(lut, C, z14[j], q0, &ak[b], &kap[j], &hold[j]);
      }
      for (int b = 0; b < 4; b++) a[gu * 4 + b] = ak[b];
      int* hsk = hs + gu * 16; int* hk = h + gu * 16; int* slk = sel + gu * 16;
      for (int c = 0; c < 16; c++) {                         // heatmap, bit 0 first, rounded after each product
        long long v = 0;
        for (int b = 0; b < 4; b++) { const long long xb = ((c >> b) & 1) ? ak[b] : C.ONE - ak[b]; v = b == 0 ? xb : hq_rne(v * xb, C.FA); }
        hsk[c] = (int)v;
      }
      long long sm = 0;                                      // top-K: rank by key 16 hs + (15 - c), all distinct
      for (int c = 0; c < 16; c++) {
        int rank = 0; const long long kc = (long long)hsk[c] * 16 + (15 - c);
        for (int c2 = 0; c2 < 16; c2++) rank += (long long)hsk[c2] * 16 + (15 - c2) > kc;
        slk[c] = rank < C.K; if (rank < C.K) sm += hsk[c];
      }
      const long long R = (1LL << 30) / (sm > 1 ? sm : 1);
      for (int c = 0; c < 16; c++) hk[c] = C.K >= 16 ? hsk[c] : (slk[c] ? (int)hq_rne((long long)hsk[c] * R, 30 - C.FA) : 0);
      if (l == 0) {
        int mix = 0;
        for (int c = 0; c < 16; c++) mix += hk[c] * f[k * 16 + c];
        const int v = (int)hq_sat(hq_rne(mix, C.FF), C.MA);
        o[k] = v; x[g.xoff[1] + k] = v;
        if (fan) x[g.xoff[2] + k] = v;
      }
    }
    HG_SYNC();
    // ---- rule stages of layers 2 and 3 (fan-out: both at once)
    for (int l0 = 1; l0 < 3; l0 += fan ? 2 : 1) {
      const int nl = fan ? 2 : 1;
      HG_FOR(idx, nl * U16) {
        const int l = l0 + idx / U16, r = idx % U16, k = r / 16, N1 = g.N1[l];
        const int* Wr = W + g.woff[l] + (long)r * N1; const int* Mk = M + g.moff[l] + (long)k * N1; const int* xl = x + g.xoff[l];
        unsigned acc = 0;
        for (int j = 0; j < N1; j++) if (Mk[j]) acc += (unsigned)(Wr[j] * xl[j]);
        f[l * U16 + r] = (int)hq_sat(hq_rne((int)acc, C.WB - 2), C.MA);
      }
      HG_SYNC();
      HG_FOR(idx, nl * U) {
        const int l = l0 + idx / U, k = idx % U, gu = l * U + k;
        int mix = 0;
        for (int c = 0; c < 16; c++) mix += h[gu * 16 + c] * f[gu * 16 + c];
        const int v = (int)hq_sat(hq_rne(mix, C.FF), C.MA);
        o[gu] = v;
        if (l == 1 && !fan) x[g.xoff[2] + k] = v;
      }
      HG_SYNC();
    }
    // ---- U1: per unit, the error signals (every thread with work forms the mixture sum itself), gate steps, latch
    // credit and the LMS scalars
    int Ssum = 0;
    for (int i = 0; i < NU; i++) Ssum += o[i];
    if (tid == 0) P[(long)net * T + smp] = (hreal)((double)Ssum / (double)((long long)NU << C.FA));
    {
      const long long yq = hq_quantise((double)Y[(long)net * T + smp], C.FA, 1 << (C.FA + 6));
      const long long eg = hq_div_rne(NU * yq - Ssum, NU), glv = hq_gl(C, expneg, t);
      HG_FOR(gu, NU) {
        const int l = gu / U, k = gu % U, lo = l == 0 ? -1 : (fan ? 0 : l - 1), N1 = g.N1[l];
        const long long e = yq - o[gu], es = hq_rne((65536 - G) * e + G * eg, 16);
        const long long egate = C.gate_err == 2 ? eg : (C.gate_err == 1 ? e : es);
        const long long ge = hq_sat32(hq_rne(glv * egate, C.FA));
        long long tot = 0, F1[4] = {0, 0, 0, 0};
        for (int c = 0; c < 16; c++) if (sel[gu * 16 + c]) {
          const long long hf = (long long)hs[gu * 16 + c] * f[gu * 16 + c];
          tot += hf; for (int b = 0; b < 4; b++) if ((c >> b) & 1) F1[b] += hf;
        }
        for (int b = 0; b < 4; b++) {
          const int j = gu * 4 + b; const long long av = a[j], num = F1[b] * (C.ONE - av) - (tot - F1[b]) * av;
          long long Gt;
          if (lat && lo >= 0) {
            const long long den = av * (C.ONE - av);
            Gt = hq_dz(C, kap[j], num, den);
            if (cred) {
              const long long G2 = hq_sat32(hq_rne(hq_sat32(hq_rne(hq_dz(C, kap[(lo * U + k) * 4 + b], num, den) * hold[j], 14)) * C.cscale, 16));
              cr[j] = (int)hq_clamp(hq_rne(ge * G2, 18), -CB, CB);
            }
          } else Gt = hq_sat32(hq_rne(12 * num, 3 * C.AB - 32));
          cbr[j] = (int)hq_clamp(hq_rne(ge * Gt, 18), -CB, CB);
        }
        long long h2 = 0, x2 = 0;
        for (int c = 0; c < 16; c++) h2 += (long long)h[gu * 16 + c] * h[gu * 16 + c];
        const int* Mk = M + g.moff[l] + (long)k * N1; const int* xl = x + g.xoff[l];
        for (int j = 0; j < N1; j++) if (Mk[j]) x2 += (long long)xl[j] * xl[j];
        const long long den = hq_rne(h2, 2 * C.FA - 18) * hq_rne(x2, 2 * C.FA - 12) + 1073742;
        const long long c2 = hq_sat32(hq_divd((Mq * es) * (1LL << (38 - C.FA)), den));
        for (int c = 0; c < 16; c++) sc[gu * 16 + c] = (int)hq_sat32(hq_rne(c2 * h[gu * 16 + c], C.FA));
      }
    }
    HG_SYNC();
    if (cred) {                                              // ---- U1b: gate steps + the credit routed from the upper layers
      HG_FOR(j, 3 * U4) {
        const int gu = j / 4, l = gu / U, k = gu % U, b = j % 4;
        int acc = 0;
        for (int up = l + 1; up < 3; up++) if ((fan ? 0 : up - 1) == l) acc += cr[(up * U + k) * 4 + b];
        cbr[j] += acc;
      }
      HG_SYNC();
    }
    // ---- U2a: hyperplane rows Hr (OLD p) and their row scales, the new p (OLD H), the rule steps, the activity
    // (the spec's mixed float32 steps)
    const bool need_A = t < tp;
    HG_FOR(idx, 3 * U4 + NU + NU * 16 + (need_A ? NU : 0)) {
      if (idx < 3 * U4) {                                    // hyperplane row j = (l * U + k) * 4 + b (mixed, float32)
        const int j = idx, gu = j / 4;
        const float cH = (float)cbr[j] * KH; const int* Hb = H + (long)j * D; const int* pk = p + (long)gu * D; float* Hrj = Hr + (long)j * D;
        float n2 = 0.0f;
        for (int i = 0; i < D; i++) {
          const float ps = (float)pk[i] * (float)s[i], hr = (float)Hb[i] + cH * ps;
          Hrj[i] = hr; n2 += hr * hr;
        }
        gsc[j] = n2 > 0.0f ? (float)(1 << C.FH) / HG_SQRT(n2) : 0.0f;   // 0: the row keeps its H
      } else if (idx < 3 * U4 + NU) {                        // p of unit gu (mixed, float32)
        const int gu = idx - 3 * U4;
        const int* pk = p + (long)gu * D; const int* H0 = H + (long)gu * 4 * D; int* pnk = pn + (long)gu * D;
        const float cP0 = (float)cbr[gu * 4] * KP, cP1 = (float)cbr[gu * 4 + 1] * KP, cP2 = (float)cbr[gu * 4 + 2] * KP, cP3 = (float)cbr[gu * 4 + 3] * KP;
        for (int i = 0; i < D; i++) {
          const float ssum = ((cP0 * (float)H0[i] + cP1 * (float)H0[D + i]) + cP2 * (float)H0[2L * D + i]) + cP3 * (float)H0[3L * D + i];
          pnk[i] = (int)hq_clamp((long long)hq_rnf((float)pk[i] + (float)s[i] * ssum), 0, 3LL << C.FP);
        }
      } else if (idx < 3 * U4 + NU + NU * 16) {              // rule row (unit gu, corner c): W' = sat(W + rnf(sW x), MW)
        const int r = idx - 3 * U4 - NU, gu = r / 16, c = r % 16, l = gu / U, k = gu % U, N1 = g.N1[l];
        const int scv = sc[r];
        if (scv != 0) {
          int* Wr = W + g.woff[l] + (long)(k * 16 + c) * N1; const int* Mk = M + g.moff[l] + (long)k * N1; const int* xl = x + g.xoff[l];
          const float sW = (float)scv * KW;
          for (int j = 0; j < N1; j++) if (Mk[j]) Wr[j] = (int)hq_sat(Wr[j] + (long long)hq_rnf(sW * (float)xl[j]), C.MW);
        }
      } else {                                               // activity (until the prune)
        const int gu = idx - 3 * U4 - NU - NU * 16, l = gu / U, k = gu % U, N1 = g.N1[l];
        int* Ak = A + g.moff[l] + (long)k * N1; const int* xl = x + g.xoff[l];
        for (int j = 0; j < N1; j++) { const long long xa = xl[j] < 0 ? -xl[j] : xl[j]; Ak[j] += (int)hq_rne(1311 * ((xa << 8) - Ak[j]), 16); }
      }
    }
    HG_SYNC();
    // ---- U2b: renormalised hyperplanes (unchanged where the row's norm is 0) and the new p
    HG_FOR(idx, 3 * U4 + NU) {
      if (idx < 3 * U4) {
        const float gr = gsc[idx];
        if (gr != 0.0f) { int* Hb = H + (long)idx * D; const float* Hrj = Hr + (long)idx * D;
          for (int i = 0; i < D; i++) Hb[i] = (int)hq_sat((long long)hq_rnf(Hrj[i] * gr), C.MW); }
      } else {
        const int gu = idx - 3 * U4; int* pk = p + (long)gu * D; const int* pnk = pn + (long)gu * D;
        for (int i = 0; i < D; i++) pk[i] = pnk[i];
      }
    }
    HG_SYNC();
    t++;
    if (t == tp) {                                           // the one-off prune: keep round(0.6 n) inputs per unit
      HG_FOR(gu, NU) {
        const int l = gu / U, k = gu % U, N1 = g.N1[l], n = N1 - 1, ntk = n <= 4 ? n : (6 * n + 5) / 10;
        int* Mk = M + g.moff[l] + (long)k * N1; const int* Ak = A + g.moff[l] + (long)k * N1; const int* Wk = W + g.woff[l] + (long)k * 16 * N1;
        int act[HQ_MAXN]; long long scr[HQ_MAXN]; int na = 0;
        for (int j = 0; j < n; j++) if (Mk[j]) act[na++] = j;
        if (na > ntk) {
          for (int q = 0; q < na; q++) {
            long long sw = 0; for (int c = 0; c < 16; c++) { const int v = Wk[(long)c * N1 + act[q]]; sw += v < 0 ? -v : v; }
            scr[q] = sw * Ak[act[q]];
          }
          for (int i = 1; i < na; i++) {                     // stable insertion sort, ascending by score
            const int v = act[i]; const long long sv = scr[i]; int q = i - 1;
            while (q >= 0 && scr[q] > sv) { act[q + 1] = act[q]; scr[q + 1] = scr[q]; q--; }
            act[q + 1] = v; scr[q + 1] = sv;
          }
          for (int i = 0; i < na - ntk; i++) Mk[act[i]] = 0;
        }
      }
      HG_SYNC();
    }
  }
  if (tid == 0) tcount[net] = t;
}
