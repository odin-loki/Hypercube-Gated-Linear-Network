// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_cli.cpp - command-line tool for HGLN. Copyright (C) Odin Loch. Licensed under the GNU AGPL v3.0 or later (see LICENSE).
//   hgln_cli parity in.bin out.bin [glr0 glr1 [competence [units [fanout [latch latch_credit gate_err glob]]]]]
//                                                          replay a stream from given initial hyperplanes (parity vs NumPy)
//   Built with -DUSE_INT (binary hgln_q): the version 5.0 integer net hgln::NetQ<float> with Params::v5() (8 units by
//   default); bench and classify as below, and
//   hgln_q parity d N [units [bits [seed]]]                 library vs the research reference of the integer
//                                                          specification (research/alg/hgln_var.hpp): bit-identical?
//   hgln_cli bench d nets nsamples [threads [units]]       speed: single thread vs a matched 3x16 MLP, then all nets in
//                                                          parallel (threads: 0 or omitted = all hardware threads)
//   hgln_cli classify train.csv test.csv passes [threads [units]]
//                                                          one-vs-rest classifier, heads trained in parallel;
//                                                          CSV rows: label,f1..fd (features scaled to about [-1,1])
//   units: units per layer, 8, 12 or 16 (default 16; Params::units)
//   hgln_cli info                                          build and CPU: SIMD kernel table in use (FORGE build)
#include "hgln.hpp"
#ifdef USE_INT
#include "hgln_q.hpp"
#include "../research/alg/hgln_var.hpp"
#endif
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#if defined(USE_FLOAT) || defined(USE_INT)
typedef float real;
#else
typedef double real;
#endif
using namespace hgln;
#ifdef USE_INT
typedef NetQ<real> NetT;
static const char* kind() { return "int16 (v5.0 integer datapath)"; }
#else
typedef Net<real> NetT;
static const char* kind() { return sizeof(real) == 4 ? "float32" : "float64"; }
#endif
static Params bench_params(double g0, double g1, int units) { Params p = NetT::default_params(g0, g1); if (units > 0) p.units = units; return p; }

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
    printf("real: %s\nsimd target: %s\n", kind(), simd_target());
#ifdef HGLN_WITH_FORGE
    printf("cpu: %s\ntables:", simd::cpu_summary().c_str());
    for (const char* const* n = simd::targets(); *n; n++) printf(" %s%s", *n, simd::table(*n) ? "" : "(unusable)");
    printf("\n");
#endif
    return 0;
  }
#ifdef USE_INT
  if (mode == "parity" && argc >= 4) {            // integer net vs the research reference QNet, same stream
    const int d = atoi(argv[2]), N = atoi(argv[3]); Params P = Params::v5(d > 16 ? 0.02 : 0.2, d > 16 ? 0.02 : 0.05);
    if (argc > 4) P.units = atoi(argv[4]);
    if (argc > 5) P.wbits = P.abits = atoi(argv[5]);
    const uint64_t seed = argc > 6 ? strtoull(argv[6], nullptr, 10) : 1;
    hv::Var V; V.uniform = true; V.U = P.units; V.hybrid = P.fanout ? 3 : 0; V.glob = P.glob; V.mu = P.mu; V.glr0 = P.glr0; V.glr1 = P.glr1;
    V.theta = P.latch; V.theta_mode = 4; V.credit = P.latch_credit > 0; V.credit_scale = P.latch_credit; V.gate_err = P.gate_err;
    V.sparse_k = P.corners; V.wbits = P.wbits; V.abits = P.abits; V.intspec = true;
    NetQ<double> a; hv::QNet r;
    try { a.init(d, seed, P); r.init(d, seed, V); } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 1; }
    Rng rng(seed + 99); std::vector<double> x(d); int bad = -1;
    for (int t = 0; t < N; t++) {
      for (auto& v : x) v = 2 * rng.uni() - 1;
      const double y = std::sin(3 * x[0]) * std::cos(2 * x[1 % d]) + 0.5 * x[2 % d] * x[3 % d];
      if (a.predict(x.data()) != r.predict(x.data()) && bad < 0) bad = t;
      a.learn(y); r.learn(y);
    }
    printf("parity hgln_q d=%d units=%d bits=%d/%d N=%d: %s\n", d, P.units, P.wbits, P.abits, N, bad < 0 ? "bit-identical to the research reference" : "DIFFERENT");
    if (bad >= 0) printf("first differing prediction: sample %d\n", bad);
    return bad < 0 ? 0 : 1;
  }
#else
  if (mode == "parity" && argc >= 4) {            // in.bin: int32 d, int32 N, H0,H1,H2 (doubles, units x 4 x (d+1) each), X (N*d), y (N)
    FILE* f = fopen(argv[2], "rb"); if (!f) { perror("in.bin"); return 1; }
    int32_t d, N; if (fread(&d, 4, 1, f) != 1 || fread(&N, 4, 1, f) != 1) return 1;
    Params P(argc > 5 ? atof(argv[4]) : 0.2, argc > 5 ? atof(argv[5]) : 0.05);
    P.competence = argc > 6 && atoi(argv[6]) != 0;
    P.units = argc > 7 ? atoi(argv[7]) : 16;
    P.fanout = argc > 8 && atoi(argv[8]) != 0;
    if (argc > 12) { P.latch = atof(argv[9]); P.latch_credit = atof(argv[10]); P.gate_err = atoi(argv[11]); P.glob = atof(argv[12]); }
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
#endif
  if (mode == "bench" && argc >= 5) {
    int d = atoi(argv[2]), nets = atoi(argv[3]), N = atoi(argv[4]), threads = argc > 5 ? atoi(argv[5]) : 0;
    const int units = argc > 6 ? atoi(argv[6]) : bench_params(0.2, 0.05, 0).units;
    if (!valid_units(units)) { fprintf(stderr, "units must be 8, 12 or 16\n"); return 1; }
    if (threads <= 0) threads = hardware_threads();
    Rng rng(88172645463325252ULL);
    std::vector<NetT> G(nets); std::vector<MLP> M(nets);
    Params prm = bench_params(0.2, 0.05, units);
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
           d, nets, units, kind(), th, tp, tm, simd_target(), threads, tb, tbp, bytes0, G[0].model_bytes(), (double)sink);
    return 0;
  }
  if (mode == "classify" && argc >= 5) {
    std::vector<real> Xtr, Xte; std::vector<int> ytr, yte; int d1, d2;
    if (!read_csv(argv[2], Xtr, ytr, d1) || !read_csv(argv[3], Xte, yte, d2) || d1 != d2) { fprintf(stderr, "bad CSV\n"); return 1; }
    int passes = atoi(argv[4]), threads = argc > 5 ? atoi(argv[5]) : 0, C = 0; for (int v : ytr) C = std::max(C, v + 1);
    Params prm = bench_params(0.02, 0.02, argc > 6 ? atoi(argv[6]) : 0);
    if (!valid_units(prm.units)) { fprintf(stderr, "units must be 8, 12 or 16\n"); return 1; }
    Classifier<real, NetT> clf; clf.init(d1, C, 1, prm);
    std::vector<int> pred(yte.size());
    for (int ep = 1; ep <= passes; ep++) {
      clf.train_batch(Xtr.data(), ytr.data(), ytr.size(), threads);   // identical to calling train() per sample, in order
      clf.predict_batch(Xte.data(), yte.size(), pred.data(), nullptr, threads);
      int ok = 0; for (size_t t = 0; t < yte.size(); t++) ok += pred[t] == yte[t];
      printf("pass %d holdout accuracy %.1f%% (%zu samples)\n", ep, 100.0 * ok / yte.size(), yte.size());
    }
    return 0;
  }
#ifdef USE_INT
  fprintf(stderr, "usage: hgln_q parity d N [units [bits [seed]]] | bench d nets nsamples [threads [units]] |\n"
                  "       classify train.csv test.csv passes [threads [units]] | info      (Params::v5(); units 8, 12 or 16; default 8)\n");
#else
  fprintf(stderr, "usage: hgln_cli parity in.bin out.bin [glr0 glr1 [competence [units [fanout [latch credit gate_err glob]]]]] |\n"
                  "       bench d nets nsamples [threads [units]] | classify train.csv test.csv passes [threads [units]] | info\n"
                  "       (units per layer: 8, 12 or 16; default 16)\n");
#endif
  return 1;
}
