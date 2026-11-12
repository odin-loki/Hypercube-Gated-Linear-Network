// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_var.hpp - experimental HGLN with switchable changes to the algorithm (research only; scalar, readable, slow).
// With the default Var{} it is the version 4.2 reference procedure (docs/TGN_impl_spec.md section 4).
// Copyright (C) Odin Loch. GNU AGPL v3.0 or later.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

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
  int theta_mode = 0;        // 0: theta for every bit; 1: only odd units latched (mixed population); 2: theta learned per bit
  double theta_lr = 0.05;
  double theta0 = 0;         // initial theta (theta_mode 2)    // learning rate of theta (theta_mode 2)
  int sparse_k = 0;          // > 0: hard corners - each unit uses only its top-k corners (weights renormalised)
  int qbits = 0;             // > 0: fixed point - W, H, p rounded (stochastically) to multiples of 2^-qbits after each update
  int hybrid = 0;            // 1: L3 reads previous-sample L2; 2: L2 reads previous-sample L1; 3: L2 and L3 both read current L1;
                             // 4: L3 reads (previous-sample L2 + current L1) / 2     // layer l > 0 reads layer l-1's outputs from the PREVIOUS sample: all rule stages independent        // update gates (p, H) only on every n-th sample
  double glr0 = 0.2, glr1 = 0.05, tau = 1000, glob = 0.75, mu = 0.3;
  int warm = 512;
};

struct Layer {
  int U, K, C, n, N1, D;     // units, hyperplanes, corners, inputs, inputs+1, gate-input size
  std::vector<double> H, p, W, M, A, m, hn;   // hn: tracked squared norms (norm_track)
  // per-sample
  std::vector<double> inp, s, a, h, f, xm, pred, comp, z;
  std::vector<double> cb_acc, dp_acc;
  std::vector<double> ap, kap, hs, th, dth;   // + per-bit dead band and d a / d theta      // latch state (previous gate bits), d a / d z factor, soft heatmap   // share_gates: gate steps from the other layers; p_shared: summed p step
  int pr(int k) const { return pshared ? 0 : k; }
  bool pshared = false, gate_from_owner = false, skip_gate = false;
  void init(int U_, int K_, int n_, int D_) {
    U = U_; K = K_; C = 1 << K; n = n_; N1 = n + 1; D = D_;
    H.assign((size_t)U * K * D, 0); p.assign((size_t)U * D, 1); W.assign((size_t)U * C * N1, 0); M.assign((size_t)U * N1, 1);
    A.assign((size_t)U * N1, 1e-3); m.assign((size_t)U * C, 0.3); hn.assign((size_t)U * K, 1);
    inp.assign(N1, 0); s.assign(D, 0); a.assign((size_t)U * K, 0); h.assign((size_t)U * C, 0); f.assign((size_t)U * C, 0);
    xm.assign((size_t)U * N1, 0); pred.assign(U, 0); comp.assign(U, 0); z.assign((size_t)U * K, 0);
    th.assign((size_t)U * K, 0); dth.assign((size_t)U * K, 0);   // th is set from Var in Net::init
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
  void forward(const double* in, const double* side, const Var& V, const Layer* owner = nullptr) {
    for (int j = 0; j < n; j++) inp[j] = in[j];
    inp[n] = 1;
    for (int i = 0; i < D; i++) s[i] = side[i];
    for (int k = 0; k < U; k++) {
      if (owner) { for (int b = 0; b < K; b++) { a[(size_t)k * K + b] = owner->a[(size_t)k * K + b]; z[(size_t)k * K + b] = owner->z[(size_t)k * K + b]; } }
      else for (int b = 0; b < K; b++) {
        const double* Hb = &H[((size_t)k * K + b) * D]; double zz = 0;
        for (int i = 0; i < D; i++) zz += Hb[i] * (V.no_p ? s[i] : p[(size_t)pr(k) * D + i] * s[i]);
        if (V.norm_track) zz /= std::sqrt(hn[(size_t)k * K + b]);
        z[(size_t)k * K + b] = zz;
        const size_t ib = (size_t)k * K + b;
        const double thv = V.theta_mode == 2 ? th[ib] : (V.theta_mode == 1 && (k & 1) == 0 ? 0.0 : V.theta);
        if (V.theta > 0 && (thv > 0 || V.theta_mode == 2)) {   // SR flip-flop with a dead band: holds its previous value when |z| < theta
          double st = 1 / (1 + std::exp(-12 * (zz - thv))), rt = 1 / (1 + std::exp(-12 * (-zz - thv))), q0 = ap[ib];
          double av = st + (1 - st - rt) * q0;
          a[ib] = std::min(std::max(av, 1e-4), 1 - 1e-4);
          kap[ib] = st * (1 - st) * (1 - q0) + rt * (1 - rt) * q0;   // d a / d (12 z) with a_prev held (one-step gradient)
          dth[ib] = 12 * (-st * (1 - st) * (1 - q0) + rt * (1 - rt) * q0);   // d a / d theta
          ap[ib] = a[ib];
        } else {
          double sg = 1 / (1 + std::exp(-12 * zz));
          a[ib] = std::min(std::max(sg, 1e-4), 1 - 1e-4);
          kap[ib] = a[ib] * (1 - a[ib]); ap[ib] = a[ib];
        }
      }
      for (int c = 0; c < C; c++) {
        double v = 1;
        for (int b = 0; b < K; b++) v *= ((c >> b) & 1) ? a[(size_t)k * K + b] : 1 - a[(size_t)k * K + b];
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
        f[(size_t)k * C + c] = sm; mix += h[(size_t)k * C + c] * sm; cm += h[(size_t)k * C + c] * m[(size_t)k * C + c];
      }
      pred[k] = std::min(std::max(mix, -2.0), 2.0);
      comp[k] = 1 / (cm + 1e-3);
    }
  }
  // mode 0: normal; 1: compute gate steps into `sink` (another layer owns the gates), no gate step; 2: owner, adds cb_acc
  void update(double y, double gl, const double* es, const Var& V, bool pruned, int mode = 0, double* sink = nullptr) {
    if (V.p_shared) dp_acc.assign(D, 0);
    for (int k = 0; k < U; k++) {
      double e = y - pred[k], esk = es[k];
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
        cb[b] = std::min(std::max(gl * esk * g, -0.05), 0.05);
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
      double c2 = (V.mu * esk) / (1e-3 + h2 * x2);
      for (int c = 0; c < C; c++) { double sc = c2 * hk[c]; double* Wc = &W[((size_t)k * C + c) * N1]; for (int j = 0; j < N1; j++) Wc[j] += sc * xk[j]; }
      if (!pruned) for (int j = 0; j < N1; j++) A[(size_t)k * N1 + j] = 0.98 * A[(size_t)k * N1 + j] + 0.02 * std::fabs(inp[j]);
    }
    if (V.p_shared && mode != 1 && !skip_gate) for (int i = 0; i < D; i++) p[i] = std::min(std::max(p[i] + dp_acc[i], 0.0), 3.0);
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

struct Net {
  int d = 0; Var V; std::vector<Layer> L; std::vector<double> u, o, r, R, us, prev; double out = 0; long t = 0; bool pruned = false; Rng qrng{12345};
  void init(int d_, uint64_t seed, const Var& v) {
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
      L[l].forward(in, sd.data(), V, (V.share_gates && l > 0) ? &L[0] : nullptr);
    }
    if (V.pipeline || V.hybrid) for (int l = 0; l < V.layers; l++) for (int k = 0; k < V.U; k++) prev[(size_t)l * V.U + k] = L[l].pred[k];
    int G = V.layers * V.U; double cs = 0;
    for (int l = 0; l < V.layers; l++) for (int k = 0; k < V.U; k++) { o[l * V.U + k] = L[l].pred[k]; r[l * V.U + k] = V.uniform ? 1.0 : L[l].comp[k]; cs += r[l * V.U + k]; }
    out = 0; for (int i = 0; i < G; i++) { r[i] /= cs; out += r[i] * o[i]; }
    return out;
  }
  void learn(double y) {
    int G = V.layers * V.U; double gl = V.glr1 + (V.glr0 - V.glr1) * std::exp(-(double)t / V.tau), sr2 = 0;
    for (int i = 0; i < G; i++) sr2 += r[i] * r[i];
    std::vector<double> es(G);
    for (int i = 0; i < G; i++) es[i] = (1 - V.glob) * (y - o[i]) + V.glob * (y - out) * r[i] / sr2;
    for (auto& l : L) l.skip_gate = V.gate_every > 1 && (t % V.gate_every) != 0;
    if (V.share_gates) {
      std::fill(L[0].cb_acc.begin(), L[0].cb_acc.end(), 0.0);
      for (int l = V.layers - 1; l > 0; l--) L[l].update(y, gl, es.data() + l * V.U, V, pruned, 1, L[0].cb_acc.data());
      L[0].update(y, gl, es.data(), V, pruned, 2);
    } else for (int l = 0; l < V.layers; l++) L[l].update(y, gl, es.data() + l * V.U, V, pruned);
    if (V.qbits > 0) for (auto& l : L) l.quantise(qrng, V.qbits);
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
