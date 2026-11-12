// hgln.hpp - Hypercube Gated Linear Network (HGLN v4.2). Header-only C++17, no dependencies.
// Stateless online learner: call predict(x) then learn(y) once per sample.
// Author: Odin Loch
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace hgln {

template <class T> static inline T clipv(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Small deterministic generator (xorshift64) so results do not depend on the platform's <random>.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed = 88172645463325252ULL) : s(seed ? seed : 88172645463325252ULL) { for (int i = 0; i < 8; i++) next(); }
  uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
  double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() { double a = uni() + 1e-300, b = uni(); return std::sqrt(-2 * std::log(a)) * std::cos(6.283185307179586 * b); }
};

// One layer: U units; each unit = 4 hyperplanes -> 16-corner heatmap -> 16 linear rules blended by the heatmap.
template <class T> struct Layer {
  static constexpr T BETA = 12;
  int U = 0, n = 0, D = 0, N1 = 0;                           // units, inputs, side-info size (d+1), inputs+bias
  std::vector<T> H, p, W, M, A, m;                           // persistent state
  std::vector<T> inp, ucp, pu, a, h, f, xm, pred, comp;      // per-sample forward cache
  std::vector<int> act; std::vector<T> keep;                 // scratch for the one-off prune
  void init(int U_, int n_, int D_) {
    U = U_; n = n_; D = D_; N1 = n + 1;
    H.assign(U * 4 * D, 0); p.assign(U * D, 1); W.assign(U * 16 * N1, 0); M.assign(U * N1, 1);
    A.assign(U * N1, (T)1e-3); m.assign(U * 16, (T)0.3);
    inp.assign(N1, 0); ucp.assign(D, 0); pu.assign(U * D, 0); a.assign(U * 4, 0); h.assign(U * 16, 0);
    f.assign(U * 16, 0); xm.assign(U * N1, 0); pred.assign(U, 0); comp.assign(U, 0);
    act.reserve(n); keep.assign(n, 0);
  }
  void random_hyperplanes(Rng& rng) {                        // unit-norm rows, last (bias) component 0
    std::vector<double> v(D);
    for (int k = 0; k < U * 4; k++) {
      double nn = 0;
      for (int i = 0; i < D - 1; i++) { v[i] = rng.normal(); nn += v[i] * v[i]; }
      v[D - 1] = 0; nn = std::sqrt(nn);
      for (int i = 0; i < D; i++) H[k * D + i] = (T)(v[i] / nn);
    }
  }
  void forward(const T* in, const T* u) {
    for (int j = 0; j < n; j++) inp[j] = in[j];
    inp[n] = 1;
    for (int i = 0; i < D; i++) ucp[i] = u[i];
    for (int k = 0; k < U; k++) {
      T* puk = &pu[k * D]; const T* pk = &p[k * D];
      for (int i = 0; i < D; i++) puk[i] = pk[i] * u[i];
      T* ak = &a[k * 4];
      for (int b = 0; b < 4; b++) {
        const T* Hb = &H[(k * 4 + b) * D]; T z = 0;
        for (int i = 0; i < D; i++) z += Hb[i] * puk[i];
        T s = 1 / (1 + std::exp(-BETA * z));
        ak[b] = clipv(s, (T)1e-4, (T)(1 - 1e-4));
      }
      T* hk = &h[k * 16];
      for (int c = 0; c < 16; c++) {
        T v = 1;
        for (int b = 0; b < 4; b++) v *= ((c >> b) & 1) ? ak[b] : (1 - ak[b]);
        hk[c] = v;
      }
      T* xk = &xm[k * N1]; const T* Mk = &M[k * N1];
      for (int j = 0; j < N1; j++) xk[j] = inp[j] * Mk[j];
      T mix = 0, cm = 0;
      for (int c = 0; c < 16; c++) {
        const T* Wc = &W[(k * 16 + c) * N1]; T s = 0;
        for (int j = 0; j < N1; j++) s += Wc[j] * xk[j];
        f[k * 16 + c] = s; mix += hk[c] * s; cm += hk[c] * m[k * 16 + c];
      }
      pred[k] = clipv(mix, (T)-2, (T)2);
      comp[k] = 1 / (cm + (T)1e-3);
    }
  }
  void update(T y, T mu, T glr, const T* es) {
    for (int k = 0; k < U; k++) {
      T e = y - pred[k], esk = es[k];
      T* hk = &h[k * 16]; T* mk = &m[k * 16];
      for (int c = 0; c < 16; c++) mk[c] += (T)0.02 * hk[c] * (e * e - mk[c]);       // competence map
      T tot = 0, F1[4] = {0, 0, 0, 0};
      for (int c = 0; c < 16; c++) {
        T hf = hk[c] * f[k * 16 + c]; tot += hf;
        for (int b = 0; b < 4; b++) if ((c >> b) & 1) F1[b] += hf;
      }
      T cb[4]; const T* ak = &a[k * 4];
      for (int b = 0; b < 4; b++) {
        T F0 = tot - F1[b], g = BETA * (F1[b] * (1 - ak[b]) - F0 * ak[b]);
        cb[b] = clipv(glr * esk * g, (T)-0.05, (T)0.05);
      }
      T* pk = &p[k * D]; const T* puk = &pu[k * D];
      for (int i = 0; i < D; i++) {                                                  // dp uses the OLD hyperplanes
        T dp = 0;
        for (int b = 0; b < 4; b++) dp += cb[b] * H[(k * 4 + b) * D + i] * ucp[i];
        pk[i] = clipv(pk[i] + dp, (T)0, (T)3);
      }
      for (int b = 0; b < 4; b++) {                                                  // hyperplane step + renormalise
        T* Hb = &H[(k * 4 + b) * D]; T nn = 0;
        for (int i = 0; i < D; i++) { Hb[i] += cb[b] * puk[i]; nn += Hb[i] * Hb[i]; }
        nn = std::sqrt(nn); for (int i = 0; i < D; i++) Hb[i] /= nn;
      }
      T h2 = 0, x2 = 0; const T* xk = &xm[k * N1];
      for (int c = 0; c < 16; c++) h2 += hk[c] * hk[c];
      for (int j = 0; j < N1; j++) x2 += xk[j] * xk[j];
      T c2 = (mu * esk) / ((T)1e-3 + h2 * x2);                                       // normalised LMS
      for (int c = 0; c < 16; c++) {
        T* Wc = &W[(k * 16 + c) * N1]; T s = c2 * hk[c];
        for (int j = 0; j < N1; j++) Wc[j] += s * xk[j];
      }
      T* Ak = &A[k * N1];
      for (int j = 0; j < N1; j++) Ak[j] = (T)0.98 * Ak[j] + (T)0.02 * std::fabs(inp[j]);
    }
  }
  // Fixed connection budget: keep round_half_even(0.6*n) inputs per unit (n<=4: dense); bias column always kept.
  void structure_step() {
    int nt = n <= 4 ? n : (int)std::nearbyint(0.6 * n);
    for (int k = 0; k < U; k++) {
      act.clear();
      for (int j = 0; j < n; j++) if (M[k * N1 + j] > 0) act.push_back(j);
      if ((int)act.size() > nt) {
        for (int j = 0; j < n; j++) {
          T s = 0; for (int c = 0; c < 16; c++) s += std::fabs(W[(k * 16 + c) * N1 + j]);
          keep[j] = (s / 16) * A[k * N1 + j];
        }
        std::stable_sort(act.begin(), act.end(), [&](int x, int y) { return keep[x] < keep[y]; });
        int drop = (int)act.size() - nt;
        for (int i = 0; i < drop; i++) M[k * N1 + act[i]] = 0;
      }
    }
  }
};

struct Params {
  double glr0 = 0.2, glr1 = 0.05;   // gate learning rate schedule start/end (8-input streams); use 0.02/0.02 for 64-input classification
  double tau = 1000;                // schedule time constant (samples)
  double glob = 0.75;               // share of each unit's error taken from the global mixture error
  double mu = 0.3;                  // rule step (normalised LMS)
  int warm = 512;                   // first prune
  Params(double g0 = 0.2, double g1 = 0.05) : glr0(g0), glr1(g1) {}
};

// The network: 3 layers x 16 units, competitive mixture over all 48 units.
template <class T> struct Net {
  int d = 0; Layer<T> L[3]; T o[48], r[48], out = 0; std::vector<T> u; long t = 0; Params P;
  void init(int d_, uint64_t seed = 1, const Params& p = Params()) {
    d = d_; P = p; t = 0; out = 0; u.assign(d + 1, 0);
    L[0].init(16, d, d + 1); L[1].init(16, 16, d + 1); L[2].init(16, 16, d + 1);
    Rng rng(seed); for (auto& l : L) l.random_hyperplanes(rng);
  }
  T predict(const T* x) {
    T nn = 1; for (int i = 0; i < d; i++) { u[i] = x[i]; nn += x[i] * x[i]; }
    u[d] = 1; nn = std::sqrt(nn); for (int i = 0; i <= d; i++) u[i] /= nn;
    L[0].forward(x, u.data()); L[1].forward(L[0].pred.data(), u.data()); L[2].forward(L[1].pred.data(), u.data());
    T cs = 0, comp[48];
    for (int l = 0; l < 3; l++) for (int k = 0; k < 16; k++) { o[l * 16 + k] = L[l].pred[k]; comp[l * 16 + k] = L[l].comp[k]; cs += comp[l * 16 + k]; }
    out = 0; for (int i = 0; i < 48; i++) { r[i] = comp[i] / cs; out += r[i] * o[i]; }
    return out;
  }
  void learn(T y) {                   // must follow predict() on the same sample
    T gl = (T)P.glr1 + (T)(P.glr0 - P.glr1) * std::exp(-(T)t / (T)P.tau), sr2 = 0, es[48];
    for (int i = 0; i < 48; i++) sr2 += r[i] * r[i];
    T eg = y - out, glob = (T)P.glob;
    for (int i = 0; i < 48; i++) es[i] = (1 - glob) * (y - o[i]) + glob * eg * r[i] / sr2;
    for (int l = 0; l < 3; l++) L[l].update(y, (T)P.mu, gl, es + 16 * l);
    t++;
    if (t >= P.warm && t % 64 == 0) for (auto& l : L) l.structure_step();
  }
};

// One-vs-rest classifier: one Net per class on +1/-1 targets.
template <class T> struct Classifier {
  int d = 0, C = 0; std::vector<Net<T>> nets; std::vector<T> score;
  void init(int d_, int classes, uint64_t seed = 1, const Params& p = Params(0.02, 0.02)) {
    d = d_; C = classes; nets.assign(C, Net<T>()); score.assign(C, 0);
    for (int k = 0; k < C; k++) nets[k].init(d, seed * 1000003ULL + k, p);
  }
  void train(const T* x, int label) {
    for (int k = 0; k < C; k++) nets[k].predict(x);
    for (int k = 0; k < C; k++) nets[k].learn(k == label ? (T)1 : (T)-1);
  }
  int predict(const T* x) {                                  // prediction only: does not change the model
    int best = 0;
    for (int k = 0; k < C; k++) { score[k] = nets[k].predict(x); if (score[k] > score[best]) best = k; }
    return best;
  }
};

}  // namespace hgln
