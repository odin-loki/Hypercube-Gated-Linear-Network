// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_var.hpp - experimental HGLN with switchable changes to the algorithm (research only; scalar, readable, slow).
// With the default Var{} it is the version 4.2 reference procedure (docs/TGN_impl_spec.md section 4).
// Copyright (C) Odin Loch. GNU AGPL v3.0 or later.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../../include/hgln_qspec.hpp"

namespace hv {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed ? seed : 88172645463325252ULL) { for (int i = 0; i < 8; i++) next(); }
  uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
  double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() { double a = uni() + 1e-300, b = uni(); return std::sqrt(-2 * std::log(a)) * std::cos(6.283185307179586 * b); }
};

struct Var {
  int K = 4;                 // hyperplanes per unit (2^K corners)
  int U = 16;                // units per layer
  int layers = 3;            // depth
  bool uniform = false;      // drop the competence map: plain average of the units
  bool no_p = false;         // drop the per-input gate weights p (gate = H . u)
  int gate_src = 0;          // 0: every layer gates on u; 1: layers >= 2 gate on their own normalised input [a;1]
  int proj = 0;              // > 0: gates see a fixed random projection of u to `proj` dims (normalised)
  bool share_gates = false; // the three layers' unit k share one set of hyperplanes and gate weights (gradients summed)
  bool p_shared = false;     // one gate-weight vector p per layer instead of per unit
  bool norm_track = false;   // track |H_b| analytically instead of a renormalisation pass (exact in real arithmetic)
  double density = 0.6;      // prune target
  int gate_every = 1;
  bool pipeline = false;
  double theta = 0;          // > 0: Schmitt-trigger flip-flop gates: set = sig(12(z-theta)), reset = sig(12(-z-theta)),
                             //      a = set + (1 - set - reset) a_prev  (theta = 0 is exactly the stateless gate)
  int gate_err = 0, rule_err = 0;  // error that drives the gate steps / the rule steps: 0 mixed es, 1 own (local), 2 global
  bool credit = false;
  double credit_scale = 1;   // strength of the routed latch credit
  double glr_scale = 1;      // multiplies the gate learning-rate schedule (glr0, glr1)       // cross-layer latch: route the held part's gradient back to the lower layer's bit
  int theta_mode = 0;        // 0: theta for every bit; 1: only odd units latched (mixed population); 2: theta learned per bit;
                             // 3: stateless settled latch a = set/(set+reset); 4: stateless cross-layer latch (previous state =
                             //    the same unit's bit in the layer this one reads from, same sample; layer 1 stateless)
  double theta_lr = 0.05;
  double theta0 = 0;         // initial theta (theta_mode 2)    // learning rate of theta (theta_mode 2)
  int sparse_k = 0;          // > 0: hard corners - each unit uses only its top-k corners (weights renormalised)
  int qbits = 0;
  int wbits = 0, abits = 0;  // > 0: integer datapath - state (W, H, p) stored as wbits-bit integers, activations as abits-bit
                             //      integers (fixed scale per array), exact integer accumulation, stochastic-rounded updates;
                             //      per-unit control scalars (error, gate steps, LMS rate) kept as 32-bit values
  bool shiftnorm = false;
  int round_mode = 0;        // integer state updates: 0 stochastic, 1 nearest, 2 nearest + error feedback, 3 floor, 4 toward zero    // integer datapath: LMS normaliser as a power-of-two shift instead of a division             // > 0: fixed point - W, H, p rounded (stochastically) to multiples of 2^-qbits after each update
  int hybrid = 0;            // 1: L3 reads previous-sample L2; 2: L2 reads previous-sample L1; 3: L2 and L3 both read current L1;
                             // 4: L3 reads (previous-sample L2 + current L1) / 2     // layer l > 0 reads layer l-1's outputs from the PREVIOUS sample: all rule stages independent        // update gates (p, H) only on every n-th sample
  double glr0 = 0.2, glr1 = 0.05, tau = 1000, glob = 0.75, mu = 0.3;
  int warm = 512;
  bool intspec = false;      // HGLN v5.0 integer specification (include/hgln_qspec.hpp), run by QNet below: wbits/abits
                             // (0 = 16), sparse_k (0 = 16 corners), hybrid = 3 (fan-out), theta with theta_mode 4
                             // (cross-layer latch), credit, gate_err, glob, mu, glr0/glr1/tau, warm. Needs uniform = 1;
                             // norm_track is irrelevant (H is renormalised exactly on every update).
  bool intspec_ginit = false; // intspec, research only: start from the float model's Gaussian hyperplanes (rounded)
                             // instead of the specification's Philox initialisation - isolates the datapath's effect
};

// Integer emulation: x -> nearest (or stochastically rounded) multiple of 2^-frac, saturated to `bits` signed bits.
// frac = bits - 1 - (integer bits of the array's range). Sums of products of such values are exact in double (< 2^53),
// i.e. they behave like an integer accumulator; rescaling by a power of two is a shift.
inline double qn(double x, int bits, int ib) {
  if (bits <= 0) return x;
  const double sc = std::ldexp(1.0, bits - 1 - ib), m = std::ldexp(1.0, bits - 1) - 1;
  return std::min(std::max(std::nearbyint(x * sc), -m), m) / sc;
}
inline double qsr(double x, int bits, int ib, Rng& r) {
  if (bits <= 0) return x;
  const double sc = std::ldexp(1.0, bits - 1 - ib), m = std::ldexp(1.0, bits - 1) - 1, y = x * sc, f = std::floor(y);
  return std::min(std::max(f + (r.uni() < y - f ? 1.0 : 0.0), -m), m) / sc;
}
struct Layer {
  int U, K, C, n, N1, D;     // units, hyperplanes, corners, inputs, inputs+1, gate-input size
  std::vector<double> H, p, W, M, A, m, hn;   // hn: tracked squared norms (norm_track)
  // per-sample
  std::vector<double> inp, s, a, h, f, xm, pred, comp, z;
  std::vector<double> cb_acc, dp_acc;
  std::vector<double> ap, kap, hs, th, dth, hold;   // hold: d a / d q = 1 - set - reset (cross-layer latch)   // + per-bit dead band and d a / d theta      // latch state (previous gate bits), d a / d z factor, soft heatmap   // share_gates: gate steps from the other layers; p_shared: summed p step
  int pr(int k) const { return pshared ? 0 : k; }
  bool pshared = false, gate_from_owner = false, skip_gate = false;
  void init(int U_, int K_, int n_, int D_) {
    U = U_; K = K_; C = 1 << K; n = n_; N1 = n + 1; D = D_;
    H.assign((size_t)U * K * D, 0); p.assign((size_t)U * D, 1); W.assign((size_t)U * C * N1, 0); M.assign((size_t)U * N1, 1);
    A.assign((size_t)U * N1, 1e-3); m.assign((size_t)U * C, 0.3); hn.assign((size_t)U * K, 1);
    inp.assign(N1, 0); s.assign(D, 0); a.assign((size_t)U * K, 0); h.assign((size_t)U * C, 0); f.assign((size_t)U * C, 0);
    xm.assign((size_t)U * N1, 0); pred.assign(U, 0); comp.assign(U, 0); z.assign((size_t)U * K, 0);
    th.assign((size_t)U * K, 0); dth.assign((size_t)U * K, 0); hold.assign((size_t)U * K, 0);   // th is set from Var in Net::init
    ap.assign((size_t)U * K, 0.5); kap.assign((size_t)U * K, 0.25); hs.assign((size_t)U * C, 0);
  }
  void random_hyperplanes(Rng& r, bool last_zero) {
    std::vector<double> v(D);
    for (int k = 0; k < U * K; k++) {
      double nn = 0;
      for (int i = 0; i < D; i++) { v[i] = (last_zero && i == D - 1) ? 0 : r.normal(); nn += v[i] * v[i]; }
      nn = std::sqrt(nn); for (int i = 0; i < D; i++) H[(size_t)k * D + i] = v[i] / nn;
    }
  }
  const double* lower_a = nullptr;
  double* credit_sink = nullptr; const double* credit_kap = nullptr;   // credit routing to the lower layer   // cross-layer latch: gate bits of the layer this one reads from (set by the net)
  void forward(const double* in, const double* side, const Var& V, const Layer* owner = nullptr) {
    for (int j = 0; j < n; j++) inp[j] = qn(in[j], V.abits, 1);
    inp[n] = 1;
    for (int i = 0; i < D; i++) s[i] = qn(side[i], V.abits, 1);
    for (int k = 0; k < U; k++) {
      if (owner) { for (int b = 0; b < K; b++) { a[(size_t)k * K + b] = owner->a[(size_t)k * K + b]; z[(size_t)k * K + b] = owner->z[(size_t)k * K + b]; } }
      else for (int b = 0; b < K; b++) {
        const double* Hb = &H[((size_t)k * K + b) * D]; double zz = 0;
        for (int i = 0; i < D; i++) zz += Hb[i] * (V.no_p ? s[i] : qn(p[(size_t)pr(k) * D + i] * s[i], V.abits, 1));
        zz = qn(zz, V.abits, 1);
        if (V.norm_track) zz /= std::sqrt(hn[(size_t)k * K + b]);
        z[(size_t)k * K + b] = zz;
        const size_t ib = (size_t)k * K + b;
        const double thv = V.theta_mode == 2 ? th[ib] : (V.theta_mode == 1 && (k & 1) == 0 ? 0.0 : V.theta);
        if (V.theta > 0 && V.theta_mode == 3) {   // settled (fixed-point) state of the SR latch: stateless
          double st = 1 / (1 + std::exp(-12 * (zz - V.theta))), rt = 1 / (1 + std::exp(-12 * (-zz - V.theta)));
          double av = st / (st + rt);
          a[ib] = std::min(std::max(av, 1e-4), 1 - 1e-4);
          kap[ib] = st * rt * (2 - st - rt) / ((st + rt) * (st + rt));   // d a / d (12 z)
          ap[ib] = a[ib];
        } else if (V.theta > 0 && V.theta_mode == 4 && !lower_a) {   // first layer of a cross-layer latch: plain gate
          double sg = 1 / (1 + std::exp(-12 * zz));
          a[ib] = std::min(std::max(sg, 1e-4), 1 - 1e-4); kap[ib] = a[ib] * (1 - a[ib]); ap[ib] = a[ib];
        } else if (V.theta > 0 && V.theta_mode == 4) {   // cross-layer latch: holds the lower layer's bit inside the band
          double st = 1 / (1 + std::exp(-12 * (zz - V.theta))), rt = 1 / (1 + std::exp(-12 * (-zz - V.theta))), q0 = lower_a[ib];
          double av = st + (1 - st - rt) * q0;
          a[ib] = std::min(std::max(av, 1e-4), 1 - 1e-4);
          kap[ib] = st * (1 - st) * (1 - q0) + rt * (1 - rt) * q0;
          hold[ib] = 1 - st - rt;
          ap[ib] = a[ib];
        } else if (V.theta > 0 && (thv > 0 || V.theta_mode == 2)) {   // SR flip-flop with a dead band: holds its previous value when |z| < theta
          double st = 1 / (1 + std::exp(-12 * (zz - thv))), rt = 1 / (1 + std::exp(-12 * (-zz - thv))), q0 = ap[ib];
          double av = st + (1 - st - rt) * q0;
          a[ib] = std::min(std::max(av, 1e-4), 1 - 1e-4);
          if (V.abits > 0) { const double g0 = std::ldexp(1.0, -(V.abits - 2)); a[ib] = std::min(std::max(qn(a[ib], V.abits, 1), g0), 1 - g0); }
          kap[ib] = st * (1 - st) * (1 - q0) + rt * (1 - rt) * q0;   // d a / d (12 z) with a_prev held (one-step gradient)
          dth[ib] = 12 * (-st * (1 - st) * (1 - q0) + rt * (1 - rt) * q0);   // d a / d theta
          ap[ib] = a[ib];
        } else {
          double sg = 1 / (1 + std::exp(-12 * zz));
          a[ib] = std::min(std::max(sg, 1e-4), 1 - 1e-4);   // integer datapath: a lookup table indexed by the quantised z
          if (V.abits > 0) { const double g0 = std::ldexp(1.0, -(V.abits - 2)); a[ib] = std::min(std::max(qn(a[ib], V.abits, 1), g0), 1 - g0); }
          kap[ib] = a[ib] * (1 - a[ib]); ap[ib] = a[ib];
        }
      }
      for (int c = 0; c < C; c++) {
        double v = 1;
        for (int b = 0; b < K; b++) v = qn(v * (((c >> b) & 1) ? a[(size_t)k * K + b] : 1 - a[(size_t)k * K + b]), V.abits, 1);
        h[(size_t)k * C + c] = v; hs[(size_t)k * C + c] = v;
      }
      if (V.sparse_k > 0 && V.sparse_k < C) {   // hard corners: keep the top-k, renormalise; others contribute nothing
        int idx[16]; for (int c = 0; c < C; c++) idx[c] = c;
        std::partial_sort(idx, idx + V.sparse_k, idx + C, [&](int x, int y) { return hs[(size_t)k * C + x] > hs[(size_t)k * C + y]; });
        double sm = 0; for (int q = 0; q < V.sparse_k; q++) sm += hs[(size_t)k * C + idx[q]];
        for (int c = 0; c < C; c++) h[(size_t)k * C + c] = 0;
        for (int q = 0; q < V.sparse_k; q++) h[(size_t)k * C + idx[q]] = hs[(size_t)k * C + idx[q]] / sm;
      }
      for (int j = 0; j < N1; j++) xm[(size_t)k * N1 + j] = inp[j] * M[(size_t)k * N1 + j];
      double mix = 0, cm = 0;
      for (int c = 0; c < C; c++) {
        double sm = 0; for (int j = 0; j < N1; j++) sm += W[((size_t)k * C + c) * N1 + j] * xm[(size_t)k * N1 + j];
        sm = qn(sm, V.abits, 3); f[(size_t)k * C + c] = sm; mix += h[(size_t)k * C + c] * sm; cm += h[(size_t)k * C + c] * m[(size_t)k * C + c];
      }
      pred[k] = qn(std::min(std::max(mix, -2.0), 2.0), V.abits, 1);
      comp[k] = 1 / (cm + 1e-3);
    }
  }
  // mode 0: normal; 1: compute gate steps into `sink` (another layer owns the gates), no gate step; 2: owner, adds cb_acc
  void update(double y, double gl, const double* es, const Var& V, bool pruned, int mode = 0, double* sink = nullptr) {
    if (V.p_shared) dp_acc.assign(D, 0);
    for (int k = 0; k < U; k++) {
      double e = y - pred[k], esk = es[k];
      const double eglob = V.glob > 0 ? (esk - (1 - V.glob) * e) / V.glob : e;
      const double eg_gate = V.gate_err == 1 ? e : V.gate_err == 2 ? eglob : esk;
      const double eg_rule = V.rule_err == 1 ? e : V.rule_err == 2 ? eglob : esk;
      double* hk = &h[(size_t)k * C]; double* fk = &f[(size_t)k * C];
      if (!V.uniform) for (int c = 0; c < C; c++) m[(size_t)k * C + c] += 0.02 * hk[c] * (e * e - m[(size_t)k * C + c]);
      double tot = 0, F1[8] = {0};
      // gate gradient from the soft heatmap; with hard corners only over the corners in use (straight-through)
      for (int c = 0; c < C; c++) { if (hk[c] == 0 && V.sparse_k > 0) continue; double hf = hs[(size_t)k * C + c] * fk[c]; tot += hf; for (int b = 0; b < K; b++) if ((c >> b) & 1) F1[b] += hf; }
      double cb[8];
      for (int b = 0; b < K; b++) {
        double ab = a[(size_t)k * K + b], g = 12 * (F1[b] * (1 - ab) - (tot - F1[b]) * ab);
        const double dm = F1[b] / ab - (tot - F1[b]) / (1 - ab);   // d mix / d a_b
        if (V.theta > 0) g = 12 * kap[(size_t)k * K + b] * dm;
        if (V.theta_mode == 2 && mode != 1 && !skip_gate) {        // learn the dead band: theta += lr * es * dmix/da * da/dtheta
          double& t0 = th[(size_t)k * K + b];
          t0 = std::min(std::max(t0 + std::min(std::max(V.theta_lr * esk * dm * dth[(size_t)k * K + b], -0.01), 0.01), 0.0), 0.5);
        }
        cb[b] = std::min(std::max(gl * eg_gate * g, -0.05), 0.05);
        if (V.credit && credit_sink) {   // the held part of this bit is the lower layer's decision: credit it there
          const size_t ib = (size_t)k * K + b;
          const double gl2 = V.credit_scale * 12 * credit_kap[ib] * dm * hold[ib];
          credit_sink[ib] += std::min(std::max(gl * eg_gate * gl2, -0.05), 0.05);
        }
      }
      if (mode == 1) { for (int b = 0; b < K; b++) sink[(size_t)k * K + b] += cb[b]; }
      if (mode == 2) for (int b = 0; b < K; b++) cb[b] += cb_acc[(size_t)k * K + b];
      if (mode != 1 && !skip_gate) {
      double* pk = &p[(size_t)pr(k) * D];
      // p step: uses the OLD (normalised) hyperplanes
      if (!V.no_p) for (int i = 0; i < D; i++) {
        double dp = 0;
        for (int bb = 0; bb < K; bb++) { double hv = H[((size_t)k * K + bb) * D + i]; if (V.norm_track) hv /= std::sqrt(hn[(size_t)k * K + bb]); dp += cb[bb] * hv * s[i]; }
        if (V.p_shared) dp_acc[i] += dp; else pk[i] = std::min(std::max(pk[i] + dp, 0.0), 3.0);
      }
      // hyperplane step with the forward pass's pu (old p), then unit length
      double pp = 0; if (V.norm_track) for (int i = 0; i < D; i++) { double pu = pu_old(k, i, V); pp += pu * pu; }
      for (int b = 0; b < K; b++) {
        double* Hb = &H[((size_t)k * K + b) * D];
        if (V.norm_track) {
          // H_raw += (cb |H_raw|) pu  <=>  H_norm <- normalise(H_norm + cb pu); |H_raw|^2 updated in closed form
          double& q = hn[(size_t)k * K + b]; double nr = std::sqrt(q), step = cb[b] * nr, hp = nr * z[(size_t)k * K + b];
          for (int i = 0; i < D; i++) Hb[i] += step * pu_old(k, i, V);
          q = q + 2 * step * hp + step * step * pp;
          if (q > 1e6 || q < 1e-6) { double nn = 0; for (int i = 0; i < D; i++) nn += Hb[i] * Hb[i]; nn = std::sqrt(nn); for (int i = 0; i < D; i++) Hb[i] /= nn; q = 1; }
        } else {
          double nn = 0;
          for (int i = 0; i < D; i++) { Hb[i] += cb[b] * pu_old(k, i, V); nn += Hb[i] * Hb[i]; }
          nn = std::sqrt(nn); for (int i = 0; i < D; i++) Hb[i] /= nn;
        }
      }
      }   // mode != 1
      double h2 = 0, x2 = 0; const double* xk = &xm[(size_t)k * N1];
      for (int c = 0; c < C; c++) h2 += hk[c] * hk[c];
      for (int j = 0; j < N1; j++) x2 += xk[j] * xk[j];
      double c2 = (V.mu * eg_rule) / (1e-3 + h2 * x2);
      if (V.shiftnorm) c2 = V.mu * esk * std::ldexp(1.0, -(int)std::ceil(std::log2(1e-3 + h2 * x2)));   // shift, no divider
      for (int c = 0; c < C; c++) { double sc = c2 * hk[c]; double* Wc = &W[((size_t)k * C + c) * N1]; for (int j = 0; j < N1; j++) Wc[j] += sc * xk[j]; }
      if (!pruned) for (int j = 0; j < N1; j++) A[(size_t)k * N1 + j] = 0.98 * A[(size_t)k * N1 + j] + 0.02 * std::fabs(inp[j]);
    }
    if (V.p_shared && mode != 1 && !skip_gate) for (int i = 0; i < D; i++) p[i] = std::min(std::max(p[i] + dp_acc[i], 0.0), 3.0);
  }
  std::vector<double> rW, rH, rp;   // round_mode 2: rounding residual carried to the next update (error feedback)
  // round_mode 0: stochastic rounding; 1: round to nearest (deterministic); 2: nearest with error feedback
  void quantise_int(Rng& r, int bits, int mode = 0) {   // integer state: W in [-8, 8), H in [-2, 2), p in [-4, 4)
    auto one = [&](std::vector<double>& v, std::vector<double>& res, int ib) {
      if (mode == 2 && res.size() != v.size()) res.assign(v.size(), 0.0);
      for (size_t i = 0; i < v.size(); i++) {
        if (mode == 0) v[i] = qsr(v[i], bits, ib, r);
        else if (mode == 1) v[i] = qn(v[i], bits, ib);
        else if (mode == 3 || mode == 4) {   // no rounding: drop the low bits (3: arithmetic shift = floor; 4: toward zero)
          const double sc = std::ldexp(1.0, bits - 1 - ib), m = std::ldexp(1.0, bits - 1) - 1, y = v[i] * sc;
          v[i] = std::min(std::max(mode == 3 ? std::floor(y) : std::trunc(y), -m), m) / sc;
        }
        else { double t = v[i] + res[i], q = qn(t, bits, ib); res[i] = t - q; v[i] = q; }
      }
    };
    one(W, rW, 3); one(H, rH, 1); one(p, rp, 2);
  }
  void quantise(Rng& r, int bits) {   // stochastic rounding to the fixed-point grid 2^-bits
    const double sc = std::ldexp(1.0, bits);
    auto qz = [&](std::vector<double>& v) { for (auto& x : v) { double y = x * sc, f = std::floor(y); x = (f + (r.uni() < y - f ? 1 : 0)) / sc; } };
    qz(W); qz(H); qz(p);
  }
  std::vector<double> pold;                      // p as it was in the forward pass (set by the net before update)
  double pu_old(int k, int i, const Var& V) const { return (V.no_p ? 1 : pold[(size_t)pr(k) * D + i]) * s[i]; }
  void prune(double density) {
    int nt = n <= 4 ? n : (int)std::nearbyint(density * n); std::vector<int> act; std::vector<double> keep(n);
    for (int k = 0; k < U; k++) {
      act.clear(); for (int j = 0; j < n; j++) if (M[(size_t)k * N1 + j] > 0) act.push_back(j);
      if ((int)act.size() <= nt) continue;
      for (int j = 0; j < n; j++) { double sm = 0; for (int c = 0; c < C; c++) sm += std::fabs(W[((size_t)k * C + c) * N1 + j]); keep[j] = sm / C * A[(size_t)k * N1 + j]; }
      std::stable_sort(act.begin(), act.end(), [&](int x, int y) { return keep[x] < keep[y]; });
      for (int i = 0; i < (int)act.size() - nt; i++) M[(size_t)k * N1 + act[i]] = 0;
    }
  }
  size_t state_values(const Var& V) const {   // persistent values that a lean implementation must keep after the prune
    int act = n <= 4 ? n : (int)std::nearbyint(V.density * n);
    const bool gates = !gate_from_owner;
    return (gates ? (size_t)U * K * D : 0) + (V.no_p || !gates ? 0 : (size_t)(pshared ? 1 : U) * D) + (size_t)U * C * (act + 1) + (V.uniform ? 0 : (size_t)U * C) + (V.norm_track ? (size_t)U * K : 0);
  }
};


// ---------------------------------------------------------------------------------------------------------------------
// QNet: the version 5.0 integer specification (include/hgln_qspec.hpp, section by section), written as plainly as
// possible in the dense reference layout. This is the reference that the library's NetQ must match bit for bit.
namespace qs = hgln::qspec;
struct QNet {
  int d = 0, U = 16, K = 16, gate_err = 0, warm = 512; bool fanout = false, latch = false, credit = false;
  int64_t cscale = 0;                                   // latch credit scale, Q16
  bool gauss_init = false;                              // Var::intspec_ginit
  double glr0 = 0.2, glr1 = 0.05, tau = 1000, glob = 0.75, mu = 0.3;
  qs::Fmt F{}; uint64_t seed = 1; int64_t t = 0; bool pruned = false; int32_t thq = 0;
  struct QL {
    int n = 0, N1 = 0, D = 0;
    std::vector<int32_t> H, p, W, M, A;               // H[(k*4+b)*D+i]  p[k*D+i]  W[(k*16+c)*N1+j]  M, A[k*N1+j]
    std::vector<int32_t> x, a, kap, hold, hs, h, f, o, acc; std::vector<char> sel;
  } L[3];
  std::vector<int32_t> s; int32_t S = 0;
  static constexpr int32_t CB = 53687091;               // 0.05 at Q30
  void init(int d_, uint64_t seed_, const Var& V) {
    if (!V.uniform) throw std::invalid_argument("intspec needs uniform=1");
    if (V.hybrid != 0 && V.hybrid != 3) throw std::invalid_argument("intspec: hybrid must be 0 or 3 (fan-out)");
    if (V.theta > 0 && V.theta_mode != 4) throw std::invalid_argument("intspec: only the cross-layer latch (theta_mode 4)");
    if (V.glr_scale != 1 || V.rule_err != 0) throw std::invalid_argument("intspec: glr_scale/rule_err unsupported");
    if (V.K != 4 || V.layers != 3 || V.density != 0.6 || V.gate_src || V.proj || V.no_p || V.share_gates || V.p_shared)
      throw std::invalid_argument("intspec: unsupported switch");
    d = d_; seed = seed_; U = V.U; K = V.sparse_k > 0 ? V.sparse_k : 16; fanout = V.hybrid == 3; latch = V.theta > 0;
    credit = V.credit && V.credit_scale > 0; cscale = qs::round_param(V.credit_scale, 16); gauss_init = V.intspec_ginit; gate_err = V.gate_err; warm = V.warm; glr0 = V.glr0; glr1 = V.glr1; tau = V.tau; glob = V.glob; mu = V.mu;
    F = qs::make_fmt(V.abits > 0 ? V.abits : 16, V.wbits > 0 ? V.wbits : 16);
    thq = qs::round_param(V.theta, F.FA); t = 0; pruned = false;
    for (int l = 0; l < 3; l++) {
      QL& q = L[l]; q.n = l == 0 ? d : U; q.N1 = q.n + 1; q.D = d + 1;
      q.H.assign((size_t)U * 4 * q.D, 0); q.p.assign((size_t)U * q.D, 1 << F.FP); q.W.assign((size_t)U * 16 * q.N1, 0);
      q.M.assign((size_t)U * q.N1, 1); q.A.assign((size_t)U * q.N1, (int32_t)qs::div_rne((int64_t)1 << (F.FA + 8), 1000));
      q.x.assign(q.N1, 0); q.a.assign(U * 4, 0); q.kap.assign(U * 4, 0); q.hold.assign(U * 4, 0); q.acc.assign(U * 4, 0);
      q.hs.assign(U * 16, 0); q.h.assign(U * 16, 0); q.f.assign(U * 16, 0); q.o.assign(U, 0); q.sel.assign(U * 16, 0);
      if (gauss_init) continue;
      for (int k = 0; k < U; k++) for (int b = 0; b < 4; b++) {
        std::vector<int32_t> v(q.D);
        for (int i = 0; i < q.D; i++) v[i] = qs::init_component(seed, l, k, b, i, q.D);
        qs::init_normalise(q.D, F, [&](int i) { return v[i]; }, [&](int i, int32_t h) { q.H[((size_t)k * 4 + b) * q.D + i] = h; });
      }
    }
    if (gauss_init) {   // research only (intspec_ginit=1): the float model's Gaussian hyperplanes, rounded to Q(WB-2)
      Rng rng(seed);
      for (int l = 0; l < 3; l++) for (int k = 0; k < U; k++) for (int b = 0; b < 4; b++) {
        std::vector<double> v(d + 1); double nn = 0;
        for (int i = 0; i <= d; i++) { v[i] = i == d ? 0 : rng.normal(); nn += v[i] * v[i]; }
        nn = std::sqrt(nn); for (int i = 0; i <= d; i++) L[l].H[((size_t)k * 4 + b) * (d + 1) + i] = (int32_t)std::nearbyint(v[i] / nn * (1 << F.FH));
      }
    }
    s.assign(d + 1, 0);
  }
  QL* lower(int l) { return l == 0 ? nullptr : (fanout ? &L[0] : &L[l - 1]); }
  double predict(const double* xin) {
    std::vector<int32_t> xq(d);
    int64_t nn = (int64_t)F.ONE * F.ONE;
    for (int j = 0; j < d; j++) { xq[j] = qs::quantise(xin[j], F.FA, F.MA); nn += (int64_t)xq[j] * xq[j]; }
    const int64_t g = (int64_t)qs::isqrt((uint64_t)nn << 6), rcp = ((int64_t)1 << (34 + F.FA)) / g;
    for (int j = 0; j < d; j++) s[j] = (int32_t)qs::rne(xq[j] * rcp, 31);
    s[d] = (int32_t)qs::rne((int64_t)F.ONE * rcp, 31);
    const int sh14 = 16 - F.AB;
    for (int l = 0; l < 3; l++) {
      QL& q = L[l]; QL* lo = lower(l);
      const int32_t* in = l == 0 ? xq.data() : lo->o.data();
      for (int j = 0; j < q.n; j++) q.x[j] = in[j];
      q.x[q.n] = F.ONE;
      for (int k = 0; k < U; k++) {
        for (int b = 0; b < 4; b++) {            // gates
          uint32_t z = 0;
          for (int i = 0; i < q.D; i++) {
            const int32_t ps = (int32_t)qs::sat(qs::rne((int64_t)q.p[(size_t)k * q.D + i] * s[i], F.FP), F.MA);
            z += (uint32_t)(q.H[((size_t)k * 4 + b) * q.D + i] * ps);
          }
          const int32_t zq = (int32_t)qs::sat(qs::rne((int32_t)z, F.FH), F.MA), z14 = zq * (1 << sh14);
          const int kb = k * 4 + b;
          int32_t a14;
          if (latch && lo) {
            const int32_t th14 = thq * (1 << sh14), st = qs::sig14(z14 - th14), rt = qs::sig14(-z14 - th14), q14 = lo->a[kb] * (1 << sh14);
            a14 = (int32_t)qs::clampi(st + qs::rne((int64_t)(16384 - st - rt) * q14, 14), 2, 16382);
            q.kap[kb] = (int32_t)(qs::rne(qs::rne((int64_t)st * (16384 - st), 14) * (16384 - q14), 14) + qs::rne(qs::rne((int64_t)rt * (16384 - rt), 14) * q14, 14));
            q.hold[kb] = 16384 - st - rt;
          } else a14 = qs::sig14(z14);
          q.a[kb] = (int32_t)qs::clampi(qs::rne(a14, sh14), 1, F.ONE - 1);
          if (!(latch && lo)) { const int64_t aq = (int64_t)q.a[kb] << sh14; q.kap[kb] = (int32_t)qs::rne(aq * (16384 - aq), 14); q.hold[kb] = 0; }
        }
        for (int c = 0; c < 16; c++) {          // heatmap, bit 0 first
          int64_t v = 0;
          for (int b = 0; b < 4; b++) {
            const int64_t xb = ((c >> b) & 1) ? q.a[k * 4 + b] : F.ONE - q.a[k * 4 + b];
            v = b == 0 ? xb : qs::rne(v * xb, F.FA);
          }
          q.hs[k * 16 + c] = (int32_t)v;
        }
        int64_t sm = 0;                          // top-K corners
        for (int c = 0; c < 16; c++) {
          int rank = 0; const int64_t kc = (int64_t)q.hs[k * 16 + c] * 16 + (15 - c);
          for (int c2 = 0; c2 < 16; c2++) rank += (int64_t)q.hs[k * 16 + c2] * 16 + (15 - c2) > kc;
          q.sel[k * 16 + c] = rank < K; if (rank < K) sm += q.hs[k * 16 + c];
        }
        const uint32_t R = (uint32_t)(((int64_t)1 << 30) / std::max<int64_t>(sm, 1));
        for (int c = 0; c < 16; c++)
          q.h[k * 16 + c] = K >= 16 ? q.hs[k * 16 + c] : (q.sel[k * 16 + c] ? (int32_t)qs::rne((int64_t)q.hs[k * 16 + c] * R, 30 - F.FA) : 0);
        int32_t mix = 0;                         // rules and mixture
        for (int c = 0; c < 16; c++) {
          uint32_t acc = 0;
          for (int j = 0; j < q.N1; j++) if (q.M[(size_t)k * q.N1 + j]) acc += (uint32_t)(q.W[((size_t)k * 16 + c) * q.N1 + j] * q.x[j]);
          q.f[k * 16 + c] = (int32_t)qs::sat(qs::rne((int32_t)acc, F.WB - 2), F.MA);
          mix += q.h[k * 16 + c] * q.f[k * 16 + c];
        }
        q.o[k] = (int32_t)qs::sat(qs::rne(mix, F.FF), F.MA);
      }
    }
    S = 0; for (int l = 0; l < 3; l++) for (int k = 0; k < U; k++) S += L[l].o[k];
    return (double)S / (double)((int64_t)3 * U << F.FA);
  }
  void learn(double y) {
    const int NU = 3 * U;
    const int64_t yq = qs::quantise(y, F.FA, 1 << (F.FA + 6));
    const int64_t eg = qs::div_rne(NU * yq - S, NU), G = qs::round_param(glob, 16), Mq = qs::round_param(mu, 16);
    const int64_t glq = qs::gl_q24(t, glr0, glr1, tau);
    for (auto& q : L) std::fill(q.acc.begin(), q.acc.end(), 0);
    for (int l = 2; l >= 0; l--) {               // upper layers first: credit reaches the lower layer in this sample
      QL& q = L[l]; QL* lo = lower(l); const int D = q.D, N1 = q.N1;
      for (int k = 0; k < U; k++) {
        const int64_t e = yq - q.o[k], es = qs::rne((65536 - G) * e + G * eg, 16);
        const int64_t egate = gate_err == 2 ? eg : (gate_err == 1 ? e : es);
        const int64_t ge = qs::sat32(qs::rne(glq * egate, F.FA));
        int64_t tot = 0, F1[4] = {0, 0, 0, 0};
        for (int c = 0; c < 16; c++) if (q.sel[k * 16 + c]) {
          const int64_t hf = (int64_t)q.hs[k * 16 + c] * q.f[k * 16 + c];
          tot += hf; for (int b = 0; b < 4; b++) if ((c >> b) & 1) F1[b] += hf;
        }
        int64_t cb[4];
        for (int b = 0; b < 4; b++) {
          const int kb = k * 4 + b; const int64_t a = q.a[kb], num = F1[b] * (F.ONE - a) - (tot - F1[b]) * a;
          int64_t Gt;
          if (latch && lo) {
            const int64_t den = a * (F.ONE - a);
            auto dz = [&](int64_t kap) {             // 12 kap (F1/a - F0/(1-a)) at Q24
              const int sh = F.FF - 10; int64_t nu = 12 * kap * num, de = den;
              if (sh >= 0) de <<= sh; else nu *= (int64_t)1 << -sh;
              return (int64_t)qs::sat32(qs::divd(nu, de));
            };
            Gt = dz(q.kap[kb]);
            if (credit) {
              const int64_t G2 = qs::sat32(qs::rne(qs::sat32(qs::rne(dz(lo->kap[kb]) * q.hold[kb], 14)) * cscale, 16));
              lo->acc[kb] += (int32_t)qs::clampi(qs::rne(ge * G2, 18), -CB, CB);
            }
          } else Gt = qs::sat32(qs::rne(12 * num, 3 * F.AB - 32));
          cb[b] = qs::clampi(qs::rne(ge * Gt, 18), -CB, CB) + q.acc[kb];
        }
        // p step (old H) and hyperplane step (old p), then renormalisation: float32 on the int16 state (the spec's
        // mixed learning steps), rounded back to the grid to nearest even
        {
          const float KP = std::ldexp(1.0f, -(29 + F.AB)), KH = std::ldexp(1.0f, -(27 + F.AB));
          float cP[4], cH[4];
          for (int b = 0; b < 4; b++) { cP[b] = (float)cb[b] * KP; cH[b] = (float)cb[b] * KH; }
          std::vector<float> Hr((size_t)4 * D); float N2[4] = {0, 0, 0, 0};
          for (int i = 0; i < D; i++) {
            float Hf[4];
            for (int b = 0; b < 4; b++) Hf[b] = (float)q.H[((size_t)k * 4 + b) * D + i];
            const float pf = (float)q.p[(size_t)k * D + i], sf = (float)s[i];
            const float ssum = ((cP[0] * Hf[0] + cP[1] * Hf[1]) + cP[2] * Hf[2]) + cP[3] * Hf[3];
            q.p[(size_t)k * D + i] = (int32_t)qs::clampi((int64_t)qs::rnf(pf + sf * ssum), 0, 3 << F.FP);
            const float ps = pf * sf;
            for (int b = 0; b < 4; b++) { const float hr = Hf[b] + cH[b] * ps; Hr[(size_t)b * D + i] = hr; N2[b] += hr * hr; }
          }
          for (int b = 0; b < 4; b++) {
            if (!(N2[b] > 0)) continue;
            const float g = (float)(1 << F.FH) / std::sqrt(N2[b]);
            for (int i = 0; i < D; i++)
              q.H[((size_t)k * 4 + b) * D + i] = (int32_t)qs::sat((int64_t)qs::rnf(Hr[(size_t)b * D + i] * g), F.MW);
          }
        }
        // rule step
        int64_t h2 = 0, x2 = 0;
        for (int c = 0; c < 16; c++) h2 += (int64_t)q.h[k * 16 + c] * q.h[k * 16 + c];
        for (int j = 0; j < N1; j++) if (q.M[(size_t)k * N1 + j]) x2 += (int64_t)q.x[j] * q.x[j];
        const int64_t den = qs::rne(h2, 2 * F.FA - 18) * qs::rne(x2, 2 * F.FA - 12) + 1073742;
        const int64_t c2 = qs::sat32(qs::divd((Mq * es) * ((int64_t)1 << (38 - F.FA)), den));
        const int FWs = 26 + F.AB - F.WB;
        const float KW = std::ldexp(1.0f, -FWs);
        for (int c = 0; c < 16; c++) {
          const int64_t sc = qs::sat32(qs::rne(c2 * q.h[k * 16 + c], F.FA));
          if (sc == 0) continue;
          const float sW = (float)sc * KW;
          for (int j = 0; j < N1; j++) {               // W' = sat(W + rnf(sW x), MW) (the mixed rule step)
            if (!q.M[(size_t)k * N1 + j]) continue;
            int32_t& w = q.W[((size_t)k * 16 + c) * N1 + j];
            w = (int32_t)qs::sat(w + (int64_t)qs::rnf(sW * (float)q.x[j]), F.MW);
          }
        }
        if (!pruned) for (int j = 0; j < N1; j++) {
          int32_t& A = q.A[(size_t)k * N1 + j];
          A += (int32_t)qs::rne(1311 * (((int64_t)std::abs(q.x[j]) << 8) - A), 16);
        }
      }
    }
    t++;
    if (!pruned && t >= warm && t % 64 == 0) {
      for (auto& q : L) {
        const int n = q.n, nt = n <= 4 ? n : (6 * n + 5) / 10;
        for (int k = 0; k < U; k++) {
          std::vector<int> act; std::vector<int64_t> sc(n, 0);
          for (int j = 0; j < n; j++) if (q.M[(size_t)k * q.N1 + j]) act.push_back(j);
          if ((int)act.size() <= nt) continue;
          for (int j : act) { int64_t sw = 0; for (int c = 0; c < 16; c++) sw += std::abs(q.W[((size_t)k * 16 + c) * q.N1 + j]); sc[j] = sw * q.A[(size_t)k * q.N1 + j]; }
          std::stable_sort(act.begin(), act.end(), [&](int x, int y) { return sc[x] < sc[y]; });
          for (int i = 0; i < (int)act.size() - nt; i++) q.M[(size_t)k * q.N1 + act[i]] = 0;
        }
      }
      pruned = true;
    }
  }
};

struct Net {
  int d = 0; Var V; std::vector<Layer> L; std::vector<double> u, o, r, R, us, prev; double out = 0; long t = 0; bool pruned = false; Rng qrng{12345};
  std::shared_ptr<QNet> q;                       // intspec: the integer specification runs instead
  void init(int d_, uint64_t seed, const Var& v) {
    q.reset(); if (v.intspec) { q = std::make_shared<QNet>(); q->init(d_, seed, v); }
    d = d_; V = v; t = 0; pruned = false; L.assign(V.layers, Layer());
    int Du = V.proj > 0 ? V.proj : d + 1;
    Rng rng(seed);
    for (int l = 0; l < V.layers; l++) {
      int n = l == 0 ? d : V.U;
      int D = (l > 0 && V.gate_src == 1) ? V.U + 1 : Du;
      L[l].init(V.U, V.K, n, D); L[l].pshared = V.p_shared; L[l].gate_from_owner = V.share_gates && l > 0;
      L[l].cb_acc.assign((size_t)V.U * V.K, 0);
      if (V.theta_mode == 2) std::fill(L[l].th.begin(), L[l].th.end(), V.theta0);
    }
    for (int l = 0; l < V.layers; l++) L[l].random_hyperplanes(rng, V.proj == 0 && !(l > 0 && V.gate_src == 1));
    if (V.proj > 0) { R.assign((size_t)V.proj * (d + 1), 0); for (auto& x : R) x = rng.normal() / std::sqrt((double)V.proj); }
    prev.assign((size_t)V.layers * V.U, 0);
    u.assign(d + 1, 0); us.assign(std::max(Du, V.U + 1), 0); o.assign((size_t)V.layers * V.U, 0); r = o;
  }
  double predict(const double* x) {
    if (q) return out = q->predict(x);
    double nn = 1; for (int i = 0; i < d; i++) { u[i] = x[i]; nn += x[i] * x[i]; }
    u[d] = 1; nn = std::sqrt(nn); for (int i = 0; i <= d; i++) u[i] /= nn;
    std::vector<double> side(u);
    if (V.proj > 0) {
      side.assign(V.proj, 0); double q = 0;
      for (int a = 0; a < V.proj; a++) { double s = 0; for (int i = 0; i <= d; i++) s += R[(size_t)a * (d + 1) + i] * u[i]; side[a] = s; q += s * s; }
      q = std::sqrt(q) + 1e-12; for (auto& s : side) s /= q;
    }
    for (int l = 0; l < V.layers; l++) {
      std::vector<double> mixin;
      const double* in = l == 0 ? x : (V.pipeline ? &prev[(size_t)(l - 1) * V.U] : L[l - 1].pred.data());
      if (V.hybrid == 1 && l == 2) in = &prev[(size_t)1 * V.U];
      if (V.hybrid == 2 && l == 1) in = &prev[0];
      if (V.hybrid == 3 && l == 2) in = L[0].pred.data();
      if (V.hybrid == 4 && l == 2) { mixin.resize(V.U); for (int k = 0; k < V.U; k++) mixin[k] = 0.5 * (prev[(size_t)V.U + k] + L[0].pred[k]); in = mixin.data(); }
      std::vector<double> sd = side;
      if (l > 0 && V.gate_src == 1) {
        sd.assign(V.U + 1, 0); double q = 1; for (int j = 0; j < V.U; j++) { sd[j] = in[j]; q += in[j] * in[j]; }
        sd[V.U] = 1; q = std::sqrt(q); for (auto& s : sd) s /= q;
      }
      L[l].pold = L[l].p;
      L[l].lower_a = nullptr;
      if (V.theta_mode == 4 && l > 0) L[l].lower_a = (V.hybrid == 3 && l == 2) ? L[0].a.data() : L[l - 1].a.data();
      L[l].forward(in, sd.data(), V, (V.share_gates && l > 0) ? &L[0] : nullptr);
    }
    if (V.pipeline || V.hybrid) for (int l = 0; l < V.layers; l++) for (int k = 0; k < V.U; k++) prev[(size_t)l * V.U + k] = L[l].pred[k];
    int G = V.layers * V.U; double cs = 0;
    for (int l = 0; l < V.layers; l++) for (int k = 0; k < V.U; k++) { o[l * V.U + k] = L[l].pred[k]; r[l * V.U + k] = V.uniform ? 1.0 : L[l].comp[k]; cs += r[l * V.U + k]; }
    out = 0; for (int i = 0; i < G; i++) { r[i] /= cs; out += r[i] * o[i]; }
    return out;
  }
  void learn(double y) {
    if (q) { q->learn(y); t++; return; }
    int G = V.layers * V.U; double gl = V.glr_scale * (V.glr1 + (V.glr0 - V.glr1) * std::exp(-(double)t / V.tau)), sr2 = 0;
    for (int i = 0; i < G; i++) sr2 += r[i] * r[i];
    std::vector<double> es(G);
    for (int i = 0; i < G; i++) es[i] = (1 - V.glob) * (y - o[i]) + V.glob * (y - out) * r[i] / sr2;
    for (auto& l : L) l.skip_gate = V.gate_every > 1 && (t % V.gate_every) != 0;
    if (V.share_gates) {
      std::fill(L[0].cb_acc.begin(), L[0].cb_acc.end(), 0.0);
      for (int l = V.layers - 1; l > 0; l--) L[l].update(y, gl, es.data() + l * V.U, V, pruned, 1, L[0].cb_acc.data());
      L[0].update(y, gl, es.data(), V, pruned, 2);
    } else if (V.credit && V.theta_mode == 4) {
      std::fill(L[0].cb_acc.begin(), L[0].cb_acc.end(), 0.0);
      for (int l = V.layers - 1; l > 0; l--) {
        Layer& low = (V.hybrid == 3 && l == 2) ? L[0] : L[l - 1];
        L[l].credit_sink = low.cb_acc.data(); L[l].credit_kap = low.kap.data();
        if (&low != &L[0]) std::fill(low.cb_acc.begin(), low.cb_acc.end(), 0.0);
      }
      for (int l = V.layers - 1; l >= 0; l--) L[l].update(y, gl, es.data() + l * V.U, V, pruned, l == 0 || !(V.hybrid == 3) ? 2 : 0);
    } else for (int l = 0; l < V.layers; l++) L[l].update(y, gl, es.data() + l * V.U, V, pruned);
    if (V.qbits > 0) for (auto& l : L) l.quantise(qrng, V.qbits);
    if (V.wbits > 0) for (auto& l : L) l.quantise_int(qrng, V.wbits, V.round_mode);
    t++;
    if (!pruned && t >= V.warm && t % 64 == 0) { for (auto& l : L) l.prune(V.density); pruned = true; }
  }
  size_t state_values() const { size_t s = 0; for (auto& l : L) s += l.state_values(V); return s; }
  // multiply-adds per sample after the prune (forward + update), counted from the procedure
  double macs() const {
    double s = 0;
    for (auto& l : L) {
      int act = (l.n <= 4 ? l.n : (int)std::nearbyint(V.density * l.n)) + 1;
      const double gm = l.gate_from_owner ? 0.0 : 1.0;
      double gate_f = gm * (double)l.U * l.K * l.D * (V.no_p ? 1 : 2);
      double gate_u = gm * (double)l.U * l.K * l.D * (V.no_p ? 3 : 6) + (V.norm_track ? 0 : (double)l.U * l.K * l.D * 2);
      gate_u /= V.gate_every;
      double rule = 2.0 * l.U * ((V.sparse_k > 0 && V.sparse_k < l.C) ? V.sparse_k : l.C) * act;
      s += gate_f + gate_u + 2 * rule + (double)l.U * l.C * (l.K + 4);
    }
    if (V.proj > 0) s += (double)V.proj * (d + 1);
    return s;
  }
};

}  // namespace hv
