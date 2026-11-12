// hgln.cpp - Hypercube Gated Linear Network (HGLN v4.1): single-thread C++17 reference port, no dependencies.
// Build:  g++ -O3 -march=native -std=c++17 hgln.cpp -o hgln_d            (float64, used for the parity test)
//         g++ -O3 -march=native -std=c++17 -DUSE_FLOAT hgln.cpp -o hgln_f (float32)
// Modes:  hgln parity in.bin out.bin [glr0 glr1] | hgln bench d nets nsamples
// Author: Odin Loch
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <string>
#ifdef USE_FLOAT
typedef float real;
#else
typedef double real;
#endif
static const real BETA = 12;
static inline real clipr(real v, real lo, real hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Layer {                       // 16 units; each unit = 4 hyperplanes -> 16-corner heatmap -> 16 linear rules
  int U, n, D, N1;                   // units, inputs, side-info size (d+1), inputs+bias
  std::vector<real> H, p, W, M, A, m;                 // learned / persistent state
  std::vector<real> inp, ucp, pu, a, h, f, xm, pred, comp;   // per-sample forward state
  void init(int U_, int n_, int D_) {
    U = U_; n = n_; D = D_; N1 = n + 1;
    H.assign(U * 4 * D, 0); p.assign(U * D, 1); W.assign(U * 16 * N1, 0); M.assign(U * N1, 1);
    A.assign(U * N1, (real)1e-3); m.assign(U * 16, (real)0.3);
    inp.assign(N1, 0); ucp.assign(D, 0); pu.assign(U * D, 0); a.assign(U * 4, 0); h.assign(U * 16, 0);
    f.assign(U * 16, 0); xm.assign(U * N1, 0); pred.assign(U, 0); comp.assign(U, 0);
  }
  void forward(const real* in, const real* u) {
    for (int j = 0; j < n; j++) inp[j] = in[j];
    inp[n] = 1;
    for (int i = 0; i < D; i++) ucp[i] = u[i];
    for (int k = 0; k < U; k++) {
      real* puk = &pu[k * D]; const real* pk = &p[k * D];
      for (int i = 0; i < D; i++) puk[i] = pk[i] * u[i];
      real* ak = &a[k * 4];
      for (int b = 0; b < 4; b++) {
        const real* Hb = &H[(k * 4 + b) * D]; real z = 0;
        for (int i = 0; i < D; i++) z += Hb[i] * puk[i];
        real s = 1 / (1 + std::exp(-BETA * z));
        ak[b] = clipr(s, (real)1e-4, (real)(1 - 1e-4));
      }
      real* hk = &h[k * 16];
      for (int c = 0; c < 16; c++) {
        real v = 1;
        for (int b = 0; b < 4; b++) v *= ((c >> b) & 1) ? ak[b] : (1 - ak[b]);
        hk[c] = v;
      }
      real* xk = &xm[k * N1]; const real* Mk = &M[k * N1];
      for (int j = 0; j < N1; j++) xk[j] = inp[j] * Mk[j];
      real mix = 0, cm = 0;
      for (int c = 0; c < 16; c++) {
        const real* Wc = &W[(k * 16 + c) * N1]; real s = 0;
        for (int j = 0; j < N1; j++) s += Wc[j] * xk[j];
        f[k * 16 + c] = s; mix += hk[c] * s; cm += hk[c] * m[k * 16 + c];
      }
      pred[k] = clipr(mix, (real)-2, (real)2);
      comp[k] = 1 / (cm + (real)1e-3);
    }
  }
  void update(real y, real mu, real glr, const real* es) {
    for (int k = 0; k < U; k++) {
      real e = y - pred[k], esk = es[k];
      real* hk = &h[k * 16]; real* mk = &m[k * 16];
      for (int c = 0; c < 16; c++) mk[c] += (real)0.02 * hk[c] * (e * e - mk[c]);   // competence map
      real tot = 0, F1[4] = {0, 0, 0, 0};
      for (int c = 0; c < 16; c++) {
        real hf = hk[c] * f[k * 16 + c]; tot += hf;
        for (int b = 0; b < 4; b++) if ((c >> b) & 1) F1[b] += hf;
      }
      real cb[4]; const real* ak = &a[k * 4];
      for (int b = 0; b < 4; b++) {                                                  // dm/dz_b
        real F0 = tot - F1[b], g = BETA * (F1[b] * (1 - ak[b]) - F0 * ak[b]);
        cb[b] = clipr(glr * esk * g, (real)-0.05, (real)0.05);
      }
      real* pk = &p[k * D]; const real* puk = &pu[k * D];
      for (int i = 0; i < D; i++) {                                                  // dp uses the OLD hyperplanes
        real dp = 0;
        for (int b = 0; b < 4; b++) dp += cb[b] * H[(k * 4 + b) * D + i] * ucp[i];
        pk[i] = clipr(pk[i] + dp, (real)0, (real)3);
      }
      for (int b = 0; b < 4; b++) {                                                  // hyperplane step + renormalise
        real* Hb = &H[(k * 4 + b) * D]; real nn = 0;
        for (int i = 0; i < D; i++) { Hb[i] += cb[b] * puk[i]; nn += Hb[i] * Hb[i]; }
        nn = std::sqrt(nn); for (int i = 0; i < D; i++) Hb[i] /= nn;
      }
      real h2 = 0, x2 = 0; const real* xk = &xm[k * N1];
      for (int c = 0; c < 16; c++) h2 += hk[c] * hk[c];
      for (int j = 0; j < N1; j++) x2 += xk[j] * xk[j];
      real c2 = (mu * esk) / ((real)1e-3 + h2 * x2);                 // normalised LMS
      for (int c = 0; c < 16; c++) {
        real* Wc = &W[(k * 16 + c) * N1]; real s = c2 * hk[c];
        for (int j = 0; j < N1; j++) Wc[j] += s * xk[j];
      }
      real* Ak = &A[k * N1];
      for (int j = 0; j < N1; j++) Ak[j] = (real)0.98 * Ak[j] + (real)0.02 * std::fabs(inp[j]);
    }
  }
  void structure_step() {            // fixed budget: keep round(0.6*n) inputs per unit (n<=4: dense); bias column always kept
    int nt = n <= 4 ? n : (int)std::nearbyint(0.6 * n);
    for (int k = 0; k < U; k++) {
      std::vector<int> act;
      for (int j = 0; j < n; j++) if (M[k * N1 + j] > 0) act.push_back(j);
      if ((int)act.size() > nt) {
        std::vector<real> keep(n, 0);
        for (int j = 0; j < n; j++) {
          real s = 0; for (int c = 0; c < 16; c++) s += std::fabs(W[(k * 16 + c) * N1 + j]);
          keep[j] = (s / 16) * A[k * N1 + j];
        }
        std::stable_sort(act.begin(), act.end(), [&](int x, int y) { return keep[x] < keep[y]; });
        int drop = (int)act.size() - nt;
        for (int i = 0; i < drop; i++) M[k * N1 + act[i]] = 0;
      }
    }
  }
};

struct HGLN {
  int d; Layer L[3]; real o[48], r[48], out, u[256]; long t; real glr0, glr1, tau, mu, glob; int warm;
  void init(int d_, real glr0_ = 0.2, real glr1_ = 0.05, real tau_ = 1000, real glob_ = 0.75) {
    d = d_; glr0 = glr0_; glr1 = glr1_; tau = tau_; mu = (real)0.3; glob = glob_; warm = 512; t = 0; out = 0;
    L[0].init(16, d, d + 1); L[1].init(16, 16, d + 1); L[2].init(16, 16, d + 1);
  }
  real predict(const real* x) {
    real nn = 1; for (int i = 0; i < d; i++) { u[i] = x[i]; nn += x[i] * x[i]; }
    u[d] = 1; nn = std::sqrt(nn); for (int i = 0; i <= d; i++) u[i] /= nn;
    L[0].forward(x, u); L[1].forward(L[0].pred.data(), u); L[2].forward(L[1].pred.data(), u);
    real cs = 0, comp[48];
    for (int l = 0; l < 3; l++) for (int k = 0; k < 16; k++) { o[l * 16 + k] = L[l].pred[k]; comp[l * 16 + k] = L[l].comp[k]; cs += comp[l * 16 + k]; }
    out = 0; for (int i = 0; i < 48; i++) { r[i] = comp[i] / cs; out += r[i] * o[i]; }
    return out;
  }
  void learn(real y) {
    real gl = glr1 + (glr0 - glr1) * std::exp(-(real)t / tau), sr2 = 0, es[48];
    for (int i = 0; i < 48; i++) sr2 += r[i] * r[i];
    real eg = y - out;
    for (int i = 0; i < 48; i++) es[i] = (1 - glob) * (y - o[i]) + glob * eg * r[i] / sr2;
    for (int l = 0; l < 3; l++) L[l].update(y, mu, gl, es + 16 * l);
    t++;
    if (t >= warm && t % 64 == 0) for (int l = 0; l < 3; l++) L[l].structure_step();
  }
};

struct Rng { uint64_t s; uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
  double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() { double a = uni() + 1e-300, b = uni(); return std::sqrt(-2 * std::log(a)) * std::cos(6.283185307179586 * b); } };

static void random_H(HGLN& g, Rng& rng) {
  for (int l = 0; l < 3; l++) { Layer& La = g.L[l];
    for (int k = 0; k < La.U * 4; k++) { double nn = 0; std::vector<double> v(La.D);
      for (int i = 0; i < La.D - 1; i++) { v[i] = rng.normal(); nn += v[i] * v[i]; } v[La.D - 1] = 0; nn = std::sqrt(nn);
      for (int i = 0; i < La.D; i++) La.H[k * La.D + i] = (real)(v[i] / nn); } }
}

struct MLP {                         // matched baseline: d -> 16 -> 16 -> 16 -> 1, tanh, per-sample SGD
  int d; std::vector<real> W[4], b[4], act[4]; real lr;
  void init(int d_, Rng& rng, real lr_ = 0.03f) { d = d_; lr = lr_; int sz[5] = {d, 16, 16, 16, 1};
    for (int i = 0; i < 4; i++) { W[i].resize(sz[i + 1] * sz[i]); b[i].assign(sz[i + 1], 0); act[i + 0].assign(sz[i + 1], 0);
      for (auto& w : W[i]) w = (real)(rng.normal() / std::sqrt((double)sz[i])); } }
  real predict(const real* x) { int sz[5] = {d, 16, 16, 16, 1}; const real* in = x; xin = in;
    for (int i = 0; i < 4; i++) { for (int o = 0; o < sz[i + 1]; o++) { real s = b[i][o]; for (int j = 0; j < sz[i]; j++) s += W[i][o * sz[i] + j] * in[j];
        act[i][o] = i < 3 ? std::tanh(s) : s; } in = act[i].data(); }
    return act[3][0]; }
  const real* xin;
  void learn(real y) { int sz[5] = {d, 16, 16, 16, 1}; std::vector<real> dl(1, act[3][0] - y);
    for (int i = 3; i >= 0; i--) { const real* in = i == 0 ? xin : act[i - 1].data(); std::vector<real> nd(sz[i], 0);
      for (int o = 0; o < sz[i + 1]; o++) { for (int j = 0; j < sz[i]; j++) { nd[j] += W[i][o * sz[i] + j] * dl[o]; W[i][o * sz[i] + j] -= lr * dl[o] * in[j]; } b[i][o] -= lr * dl[o]; }
      if (i > 0) for (int j = 0; j < sz[i]; j++) nd[j] *= (1 - act[i - 1][j] * act[i - 1][j]);
      dl = nd; } }
};

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "parity") {                       // in.bin: int32 d, int32 N, H0,H1,H2 (doubles), X (N*d), y (N)
    FILE* f = fopen(argv[2], "rb"); int32_t d, N; fread(&d, 4, 1, f); fread(&N, 4, 1, f);
    HGLN g; if (argc > 5) g.init(d, (real)atof(argv[4]), (real)atof(argv[5])); else g.init(d);
    for (int l = 0; l < 3; l++) { std::vector<double> buf(g.L[l].H.size()); fread(buf.data(), 8, buf.size(), f); for (size_t i = 0; i < buf.size(); i++) g.L[l].H[i] = (real)buf[i]; }
    std::vector<double> X((size_t)N * d), Y(N); fread(X.data(), 8, X.size(), f); fread(Y.data(), 8, Y.size(), f); fclose(f);
    std::vector<double> P(N); std::vector<real> x(d);
    for (int t = 0; t < N; t++) { for (int i = 0; i < d; i++) x[i] = (real)X[(size_t)t * d + i]; P[t] = g.predict(x.data()); g.learn((real)Y[t]); }
    f = fopen(argv[3], "wb"); fwrite(P.data(), 8, N, f); fclose(f); return 0;
  }
  if (mode == "bench") {                        // bench d nets nsamples
    int d = atoi(argv[2]), nets = atoi(argv[3]), N = atoi(argv[4]); Rng rng{88172645463325252ULL};
    std::vector<HGLN> G(nets); std::vector<MLP> M(nets);
    for (int k = 0; k < nets; k++) { G[k].init(d); random_H(G[k], rng); M[k].init(d, rng); }
    std::vector<real> x(d); auto gen = [&](real& y) { for (int i = 0; i < d; i++) x[i] = (real)(2 * rng.uni() - 1);
      y = (real)(std::sin(3 * x[0]) * std::cos(2 * x[1]) + 0.5 * x[2 % d] * x[3 % d]); };
    real y, sink = 0; int warm = 1500;
    for (int t = 0; t < warm; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += G[k].predict(x.data()); G[k].learn(y); } }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += G[k].predict(x.data()); G[k].learn(y); } }
    double th = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count() / N;
    t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) sink += G[k].predict(x.data()); }
    double tp = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count() / N;
    t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += M[k].predict(x.data()); M[k].learn(y); } }
    double tm = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count() / N;
    printf("{\"d\":%d,\"nets\":%d,\"real\":\"%s\",\"hgln_us_per_sample\":%.2f,\"hgln_predict_only_us\":%.2f,\"mlp3x16_us_per_sample\":%.2f,\"sink\":%g}\n",
           d, nets, sizeof(real) == 4 ? "float32" : "float64", th, tp, tm, (double)sink);
    return 0;
  }
  fprintf(stderr, "usage: hgln parity in.bin out.bin [glr0 glr1] | hgln bench d nets nsamples\n"); return 1;
}
