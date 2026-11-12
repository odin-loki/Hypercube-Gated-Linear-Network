// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_cli.cpp - command-line tool for HGLN. Copyright (C) Odin Loch. Licensed under the GNU AGPL v3.0 or later (see LICENSE).
//   hgln_cli parity in.bin out.bin [glr0 glr1 [competence [units [fanout]]]]
//                                                          replay a stream from given initial hyperplanes (parity vs NumPy)
//   hgln_cli bench d nets nsamples [threads [units]]       speed: single thread vs a matched 3x16 MLP, then all nets in
//                                                          parallel (threads: 0 or omitted = all hardware threads)
//   hgln_cli classify train.csv test.csv passes [threads [units]]
//                                                          one-vs-rest classifier, heads trained in parallel;
//                                                          CSV rows: label,f1..fd (features scaled to about [-1,1])
//   units: units per layer, 8, 12 or 16 (default 16; Params::units)
//   hgln_cli info                                          build and CPU: SIMD kernel table in use (FORGE build)
#include "hgln.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#ifdef USE_FLOAT
typedef float real;
#else
typedef double real;
#endif
using namespace hgln;

struct MLP {                         // matched baseline: d -> 16 -> 16 -> 16 -> 1, tanh, per-sample SGD
  int d; std::vector<real> W[4], b[4], act[4]; real lr; const real* xin;
  void init(int d_, Rng& rng, real lr_ = 0.03f) { d = d_; lr = lr_; int sz[5] = {d, 16, 16, 16, 1};
    for (int i = 0; i < 4; i++) { W[i].resize(sz[i + 1] * sz[i]); b[i].assign(sz[i + 1], 0); act[i].assign(sz[i + 1], 0);
      for (auto& w : W[i]) w = (real)(rng.normal() / std::sqrt((double)sz[i])); } }
  real predict(const real* x) { int sz[5] = {d, 16, 16, 16, 1}; const real* in = x; xin = in;
    for (int i = 0; i < 4; i++) { for (int o = 0; o < sz[i + 1]; o++) { real s = b[i][o]; for (int j = 0; j < sz[i]; j++) s += W[i][o * sz[i] + j] * in[j];
        act[i][o] = i < 3 ? std::tanh(s) : s; } in = act[i].data(); }
    return act[3][0]; }
  void learn(real y) { int sz[5] = {d, 16, 16, 16, 1}; std::vector<real> dl(1, act[3][0] - y);
    for (int i = 3; i >= 0; i--) { const real* in = i == 0 ? xin : act[i - 1].data(); std::vector<real> nd(sz[i], 0);
      for (int o = 0; o < sz[i + 1]; o++) { for (int j = 0; j < sz[i]; j++) { nd[j] += W[i][o * sz[i] + j] * dl[o]; W[i][o * sz[i] + j] -= lr * dl[o] * in[j]; } b[i][o] -= lr * dl[o]; }
      if (i > 0) for (int j = 0; j < sz[i]; j++) nd[j] *= (1 - act[i - 1][j] * act[i - 1][j]);
      dl = nd; } }
};

static bool read_csv(const char* path, std::vector<real>& X, std::vector<int>& y, int& d) {
  std::ifstream f(path); if (!f) return false; std::string line; d = -1;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line); std::string tok; std::vector<double> v;
    while (std::getline(ss, tok, ',')) v.push_back(atof(tok.c_str()));
    if (v.size() < 2) return false;
    if (d < 0) d = (int)v.size() - 1;
    if ((int)v.size() - 1 != d) return false;
    y.push_back((int)v[0]); for (int i = 1; i <= d; i++) X.push_back((real)v[i]);
  }
  return d > 0;
}

#ifdef HGLN_WITH_FORGE
static const char* simd_target() { return simd::active().target; }   // FORGE_SIMD_TARGET=scalar|avx2|avx512|... forces one
#else
static const char* simd_target() { return "header"; }               // header-only build: kernels inlined at compile time
#endif

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "info") {
    printf("real: %s\nsimd target: %s\n", sizeof(real) == 4 ? "float32" : "float64", simd_target());
#ifdef HGLN_WITH_FORGE
    printf("cpu: %s\ntables:", simd::cpu_summary().c_str());
    for (const char* const* n = simd::targets(); *n; n++) printf(" %s%s", *n, simd::table(*n) ? "" : "(unusable)");
    printf("\n");
#endif
    return 0;
  }
  if (mode == "parity" && argc >= 4) {            // in.bin: int32 d, int32 N, H0,H1,H2 (doubles, units x 4 x (d+1) each), X (N*d), y (N)
    FILE* f = fopen(argv[2], "rb"); if (!f) { perror("in.bin"); return 1; }
    int32_t d, N; if (fread(&d, 4, 1, f) != 1 || fread(&N, 4, 1, f) != 1) return 1;
    Params P(argc > 5 ? atof(argv[4]) : 0.2, argc > 5 ? atof(argv[5]) : 0.05);
    P.competence = argc > 6 && atoi(argv[6]) != 0;
    P.units = argc > 7 ? atoi(argv[7]) : 16;
    P.fanout = argc > 8 && atoi(argv[8]) != 0;
    if (!valid_units(P.units)) { fprintf(stderr, "units must be 8, 12 or 16\n"); return 1; }
    Net<real> g; g.init(d, 1, P);
    for (int l = 0; l < 3; l++) { std::vector<double> buf((size_t)g.L[l].U * 4 * g.L[l].D);
      if (fread(buf.data(), 8, buf.size(), f) != buf.size()) return 1;
      g.L[l].set_hyperplanes(buf.data()); }
    std::vector<double> X((size_t)N * d), Y(N);
    if (fread(X.data(), 8, X.size(), f) != X.size() || fread(Y.data(), 8, Y.size(), f) != Y.size()) return 1;
    fclose(f);
    std::vector<double> Pd(N); std::vector<real> x(d);
    for (int t = 0; t < N; t++) { for (int i = 0; i < d; i++) x[i] = (real)X[(size_t)t * d + i]; Pd[t] = g.predict(x.data()); g.learn((real)Y[t]); }
    f = fopen(argv[3], "wb"); fwrite(Pd.data(), 8, N, f); fclose(f); return 0;
  }
  if (mode == "bench" && argc >= 5) {
    int d = atoi(argv[2]), nets = atoi(argv[3]), N = atoi(argv[4]), threads = argc > 5 ? atoi(argv[5]) : 0;
    const int units = argc > 6 ? atoi(argv[6]) : 16;
    if (!valid_units(units)) { fprintf(stderr, "units must be 8, 12 or 16\n"); return 1; }
    if (threads <= 0) threads = hardware_threads();
    Rng rng(88172645463325252ULL);
    std::vector<Net<real>> G(nets); std::vector<MLP> M(nets);
    Params prm; prm.units = units;
    for (int k = 0; k < nets; k++) { G[k].init(d, 100 + k, prm); M[k].init(d, rng); }
    size_t bytes0 = G[0].model_bytes();
    std::vector<real> x(d); auto gen = [&](real& y) { for (int i = 0; i < d; i++) x[i] = (real)(2 * rng.uni() - 1);
      y = (real)(std::sin(3 * x[0]) * std::cos(2 * x[1]) + 0.5 * x[2 % d] * x[3 % d]); };
    real y, sink = 0; int warm = 1500;
    for (int t = 0; t < warm; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += G[k].predict(x.data()); G[k].learn(y); } }
    auto now = [] { return std::chrono::high_resolution_clock::now(); };
    auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
    // 1) single thread, one sample at a time (the original protocol)
    auto t0 = now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += G[k].predict(x.data()); G[k].learn(y); } }
    double th = us(t0, now()) / N;
    t0 = now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) sink += G[k].predict(x.data()); }
    double tp = us(t0, now()) / N;
    t0 = now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += M[k].predict(x.data()); M[k].learn(y); } }
    double tm = us(t0, now()) / N;
    // 2) batched: the same N samples, all networks in parallel (each network still learns sample by sample)
    std::vector<real> X((size_t)N * d), Y((size_t)nets * N), P((size_t)nets * N);
    for (int t = 0; t < N; t++) { gen(y); std::copy(x.begin(), x.end(), X.begin() + (size_t)t * d); for (int k = 0; k < nets; k++) Y[(size_t)k * N + t] = y; }
    t0 = now(); run_streams(G, X.data(), Y.data(), P.data(), N, threads); double tb = us(t0, now()) / N;
    t0 = now(); run_streams(G, X.data(), (const real*)nullptr, P.data(), N, threads); double tbp = us(t0, now()) / N;
    for (real v : P) sink += v;
    printf("{\"d\":%d,\"nets\":%d,\"units\":%d,\"real\":\"%s\",\"hgln_us_per_sample\":%.2f,\"hgln_predict_only_us\":%.2f,\"mlp3x16_us_per_sample\":%.2f,"
           "\"simd\":\"%s\",\"threads\":%d,\"batched_us_per_sample\":%.2f,\"batched_predict_only_us\":%.2f,"
           "\"model_bytes_per_net_initial\":%zu,\"model_bytes_per_net_after_prune\":%zu,\"sink\":%g}\n",
           d, nets, units, sizeof(real) == 4 ? "float32" : "float64", th, tp, tm, simd_target(), threads, tb, tbp, bytes0, G[0].model_bytes(), (double)sink);
    return 0;
  }
  if (mode == "classify" && argc >= 5) {
    std::vector<real> Xtr, Xte; std::vector<int> ytr, yte; int d1, d2;
    if (!read_csv(argv[2], Xtr, ytr, d1) || !read_csv(argv[3], Xte, yte, d2) || d1 != d2) { fprintf(stderr, "bad CSV\n"); return 1; }
    int passes = atoi(argv[4]), threads = argc > 5 ? atoi(argv[5]) : 0, C = 0; for (int v : ytr) C = std::max(C, v + 1);
    Params prm(0.02, 0.02); prm.units = argc > 6 ? atoi(argv[6]) : 16;
    if (!valid_units(prm.units)) { fprintf(stderr, "units must be 8, 12 or 16\n"); return 1; }
    Classifier<real> clf; clf.init(d1, C, 1, prm);
    std::vector<int> pred(yte.size());
    for (int ep = 1; ep <= passes; ep++) {
      clf.train_batch(Xtr.data(), ytr.data(), ytr.size(), threads);   // identical to calling train() per sample, in order
      clf.predict_batch(Xte.data(), yte.size(), pred.data(), nullptr, threads);
      int ok = 0; for (size_t t = 0; t < yte.size(); t++) ok += pred[t] == yte[t];
      printf("pass %d holdout accuracy %.1f%% (%zu samples)\n", ep, 100.0 * ok / yte.size(), yte.size());
    }
    return 0;
  }
  fprintf(stderr, "usage: hgln_cli parity in.bin out.bin [glr0 glr1 [competence [units [fanout]]]] | bench d nets nsamples [threads [units]] |\n"
                  "       classify train.csv test.csv passes [threads [units]] | info      (units per layer: 8, 12 or 16; default 16)\n");
  return 1;
}
