// hgln_cli.cpp - command-line tool for HGLN.
//   hgln_cli parity in.bin out.bin [glr0 glr1]   replay a stream from given initial hyperplanes (parity test vs NumPy)
//   hgln_cli bench d nets nsamples               speed benchmark against a matched 3x16 MLP
//   hgln_cli classify train.csv test.csv passes  one-vs-rest classifier; CSV rows: label,f1..fd (features scaled to about [-1,1])
// Author: Odin Loch
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

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "parity" && argc >= 4) {            // in.bin: int32 d, int32 N, H0,H1,H2 (doubles, 16x4x(d+1) each), X (N*d), y (N)
    FILE* f = fopen(argv[2], "rb"); if (!f) { perror("in.bin"); return 1; }
    int32_t d, N; if (fread(&d, 4, 1, f) != 1 || fread(&N, 4, 1, f) != 1) return 1;
    Params P(argc > 5 ? atof(argv[4]) : 0.2, argc > 5 ? atof(argv[5]) : 0.05);
    Net<real> g; g.init(d, 1, P);
    for (int l = 0; l < 3; l++) { std::vector<double> buf(g.L[l].H.size());
      if (fread(buf.data(), 8, buf.size(), f) != buf.size()) return 1;
      for (size_t i = 0; i < buf.size(); i++) g.L[l].H[i] = (real)buf[i]; }
    std::vector<double> X((size_t)N * d), Y(N);
    if (fread(X.data(), 8, X.size(), f) != X.size() || fread(Y.data(), 8, Y.size(), f) != Y.size()) return 1;
    fclose(f);
    std::vector<double> Pd(N); std::vector<real> x(d);
    for (int t = 0; t < N; t++) { for (int i = 0; i < d; i++) x[i] = (real)X[(size_t)t * d + i]; Pd[t] = g.predict(x.data()); g.learn((real)Y[t]); }
    f = fopen(argv[3], "wb"); fwrite(Pd.data(), 8, N, f); fclose(f); return 0;
  }
  if (mode == "bench" && argc >= 5) {
    int d = atoi(argv[2]), nets = atoi(argv[3]), N = atoi(argv[4]); Rng rng(88172645463325252ULL);
    std::vector<Net<real>> G(nets); std::vector<MLP> M(nets);
    for (int k = 0; k < nets; k++) { G[k].init(d, 100 + k); M[k].init(d, rng); }
    std::vector<real> x(d); auto gen = [&](real& y) { for (int i = 0; i < d; i++) x[i] = (real)(2 * rng.uni() - 1);
      y = (real)(std::sin(3 * x[0]) * std::cos(2 * x[1]) + 0.5 * x[2 % d] * x[3 % d]); };
    real y, sink = 0; int warm = 1500;
    for (int t = 0; t < warm; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += G[k].predict(x.data()); G[k].learn(y); } }
    auto now = [] { return std::chrono::high_resolution_clock::now(); };
    auto t0 = now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += G[k].predict(x.data()); G[k].learn(y); } }
    double th = std::chrono::duration<double, std::micro>(now() - t0).count() / N;
    t0 = now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) sink += G[k].predict(x.data()); }
    double tp = std::chrono::duration<double, std::micro>(now() - t0).count() / N;
    t0 = now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += M[k].predict(x.data()); M[k].learn(y); } }
    double tm = std::chrono::duration<double, std::micro>(now() - t0).count() / N;
    printf("{\"d\":%d,\"nets\":%d,\"real\":\"%s\",\"hgln_us_per_sample\":%.2f,\"hgln_predict_only_us\":%.2f,\"mlp3x16_us_per_sample\":%.2f,\"sink\":%g}\n",
           d, nets, sizeof(real) == 4 ? "float32" : "float64", th, tp, tm, (double)sink);
    return 0;
  }
  if (mode == "classify" && argc >= 5) {
    std::vector<real> Xtr, Xte; std::vector<int> ytr, yte; int d1, d2;
    if (!read_csv(argv[2], Xtr, ytr, d1) || !read_csv(argv[3], Xte, yte, d2) || d1 != d2) { fprintf(stderr, "bad CSV\n"); return 1; }
    int passes = atoi(argv[4]), C = 0; for (int v : ytr) C = std::max(C, v + 1);
    Classifier<real> clf; clf.init(d1, C, 1);
    for (int ep = 1; ep <= passes; ep++) {
      for (size_t t = 0; t < ytr.size(); t++) clf.train(&Xtr[t * d1], ytr[t]);
      int ok = 0; for (size_t t = 0; t < yte.size(); t++) ok += clf.predict(&Xte[t * d1]) == yte[t];
      printf("pass %d holdout accuracy %.1f%% (%zu samples)\n", ep, 100.0 * ok / yte.size(), yte.size());
    }
    return 0;
  }
  fprintf(stderr, "usage: hgln_cli parity in.bin out.bin [glr0 glr1] | bench d nets nsamples | classify train.csv test.csv passes\n");
  return 1;
}
