// SPDX-License-Identifier: AGPL-3.0-or-later
// test_api.cpp - behavioural tests for the header-only library. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#include "hgln.hpp"
#include <cstdio>
using namespace hgln;
static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else printf("ok:   %s\n", msg); } while (0)

template <class T> double run_stream(Net<T>& n, int d, int N, uint64_t seed, double* late_nmse, std::vector<T>* outs = nullptr) {
  Rng rng(seed); std::vector<T> x(d); double se = 0, sy = 0, sy2 = 0; int cnt = 0;
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < d; i++) x[i] = (T)(2 * rng.uni() - 1);
    T y = (T)(std::sin(3 * x[0]) * std::cos(2 * x[1]) + 0.5 * x[2] * x[3]);
    T p = n.predict(x.data()); n.learn(y); if (outs) outs->push_back(p);
    if (t >= N - 1000) { se += (y - p) * (y - p); sy += y; sy2 += (double)y * y; cnt++; }
  }
  double var = sy2 / cnt - (sy / cnt) * (sy / cnt); *late_nmse = se / cnt / var; return *late_nmse;
}
int main() {
  { Net<double> a, b; a.init(8, 5); b.init(8, 5); double n1, n2; std::vector<double> o1, o2;
    run_stream(a, 8, 1500, 9, &n1, &o1); run_stream(b, 8, 1500, 9, &n2, &o2);
    CHECK(o1 == o2, "same seed and stream give identical predictions (deterministic)"); }
  { Net<double> a, b; a.init(8, 5); b.init(8, 6); double n1, n2; std::vector<double> o1, o2;
    run_stream(a, 8, 300, 9, &n1, &o1); run_stream(b, 8, 300, 9, &n2, &o2);
    CHECK(o1 != o2, "different seeds give different networks"); }
  { Net<double> a; a.init(8, 1); double nm; run_stream(a, 8, 5000, 11, &nm);
    printf("      late NMSE after 5000 samples (d=8, smooth target): %.3f\n", nm);
    CHECK(nm < 0.35, "learns a smooth 8-input target online (late NMSE < 0.35; predicting the mean gives 1.0)"); }
  { Net<float> a; a.init(8, 1); double nm; run_stream(a, 8, 5000, 11, &nm);
    printf("      float32 late NMSE: %.3f\n", nm); CHECK(nm < 0.35, "float32 build learns too"); }
  { Net<double> a; a.init(8, 1); double nm; run_stream(a, 8, 1500, 3, &nm);
    int pruned = 0, total = 0; for (int l = 0; l < 3; l++) { for (int k = 0; k < 16; k++) { int act = 0; for (int j = 0; j < a.L[l].n; j++) act += a.L[l].active(k, j); total++; int want = (int)std::nearbyint(0.6 * a.L[l].n); pruned += act == want; if (!a.L[l].active(k, a.L[l].n)) pruned -= 1000; } }
    CHECK(pruned == total, "after warm-up every unit keeps exactly round(0.6 n) inputs and its bias column"); }
  { Net<double> a; a.init(8, 1); bool ok = true; Rng rng(5); std::vector<double> x(8);
    for (int t = 0; t < 2000; t++) { for (auto& v : x) v = 2 * rng.uni() - 1; double p = a.predict(x.data()); a.learn(100 * std::sin(10 * x[0]));
      if (!(p == p) || std::fabs(p) > 2.0 + 1e-12) ok = false;
      for (int l = 0; l < 3; l++) for (double v : a.L[l].W) if (!(v == v)) ok = false; }
    CHECK(ok, "stays finite and bounded (|prediction| <= 2) under a hostile, large-amplitude target"); }
  { Net<double> a; a.init(8, 1); std::vector<double> x(8, 1e6); bool ok = true;
    for (int t = 0; t < 200; t++) { double p = a.predict(x.data()); a.learn(1.0); if (!(p == p)) ok = false; }
    CHECK(ok, "stays finite for a very large input (1e6)"); }
  { Net<double> a; a.init(8, 1); std::vector<double> x(8, 0.0); bool ok = true;
    for (int t = 0; t < 200; t++) { double p = a.predict(x.data()); a.learn(0.0); if (!(p == p)) ok = false; }
    CHECK(ok, "stays finite for all-zero input and target"); }
  { Classifier<double> c; c.init(2, 2, 1); Rng rng(2); double x[2]; int ok = 0, n = 0;
    for (int t = 0; t < 4000; t++) { x[0] = 2 * rng.uni() - 1; x[1] = 2 * rng.uni() - 1; int lab = (x[0] * x[1] > 0);
      if (t >= 3000) { ok += c.predict(x) == lab; n++; } c.train(x, lab); }
    printf("      XOR-style 2-class accuracy on the last 1000 samples (test-then-train): %.1f%%\n", 100.0 * ok / n);
    CHECK(ok > 0.8 * n, "classifier learns an XOR-style problem (> 80%)"); }
  // ---- parallel and batched APIs must give exactly the serial results
  { Rng rng(21); const int d = 16, N = 1500, C = 4; std::vector<double> X((size_t)N * d); std::vector<int> lab(N);
    for (int t = 0; t < N; t++) { double s = 0; for (int i = 0; i < d; i++) { X[(size_t)t * d + i] = 2 * rng.uni() - 1; s += X[(size_t)t * d + i] * (i + 1); } lab[t] = ((int)std::floor(s * 3) % C + C) % C; }
    Classifier<double> a, b, c; a.init(d, C, 3); b.init(d, C, 3); c.init(d, C, 3); c.set_threads(4);
    for (int t = 0; t < N; t++) { a.train(&X[(size_t)t * d], lab[t]); c.train(&X[(size_t)t * d], lab[t]); }
    b.train_batch(X.data(), lab.data(), N, 4);
    bool same = true;
    for (int k = 0; k < C; k++) for (int l = 0; l < 3; l++) { if (!(a.nets[k].L[l].W.size() == b.nets[k].L[l].W.size())) same = false;
      for (size_t i = 0; i < a.nets[k].L[l].W.size() && same; i++) same = a.nets[k].L[l].W[i] == b.nets[k].L[l].W[i] && a.nets[k].L[l].W[i] == c.nets[k].L[l].W[i];
      for (size_t i = 0; i < a.nets[k].L[l].H.size() && same; i++) same = a.nets[k].L[l].H[i] == b.nets[k].L[l].H[i] && a.nets[k].L[l].H[i] == c.nets[k].L[l].H[i]; }
    CHECK(same, "train_batch (4 threads) and pooled train() give bit-identical models to serial train()");
    std::vector<int> pb(N); std::vector<double> sc((size_t)N * C); b.predict_batch(X.data(), N, pb.data(), sc.data(), 4);
    bool samep = true; for (int t = 0; t < N; t++) { int p1 = a.predict(&X[(size_t)t * d]), p2 = c.predict(&X[(size_t)t * d]); if (p1 != pb[t] || p2 != p1 || sc[(size_t)t * C + p1] != a.score[p1]) samep = false; }
    CHECK(samep, "predict_batch (4 threads) and pooled predict() match serial predict()");
    std::vector<Net<double>> G(3), R; for (int k = 0; k < 3; k++) G[k].init(d, 50 + k); R = G;
    std::vector<double> Y((size_t)3 * N), P((size_t)3 * N); for (size_t i = 0; i < Y.size(); i++) Y[i] = std::sin(0.01 * (double)i);
    run_streams(G, X.data(), Y.data(), P.data(), N, 3);
    bool samer = true; for (int k = 0; k < 3; k++) for (int t = 0; t < N; t++) { double v = R[k].predict(&X[(size_t)t * d]); R[k].learn(Y[(size_t)k * N + t]); if (v != P[(size_t)k * N + t]) samer = false; }
    CHECK(samer, "run_streams (3 threads) matches per-network serial predict/learn"); }
  { Net<double> a; a.init(8, 1); double nm; run_stream(a, 8, 1000, 4, &nm); Net<double> b = a; Work<double> w; w.init(a);
    std::vector<double> x(8, 0.3); double p1 = a.infer(x.data(), w), p2 = b.predict(x.data()), p3 = a.infer(x.data(), w);
    CHECK(p1 == p2 && p1 == p3, "copied network and const infer() give the same prediction; infer() leaves the model unchanged");
    Net<double> fresh; fresh.init(8, 1); size_t before = fresh.model_bytes();
    printf("      model bytes d=8: %zu before prune, %zu after\n", before, a.model_bytes());
    CHECK(a.pruned() && !fresh.pruned() && a.model_bytes() < before, "the prune compacts W and frees the activity trace (model shrinks)"); }
  { // import_dense(export) round trip, before and after the prune, continues bit-identically
    for (int Nwarm : {300, 900}) {
      Net<double> a; a.init(8, 2); double nm; run_stream(a, 8, Nwarm, 6, &nm); Net<double> b; b.init(8, 99);
      for (int l = 0; l < 3; l++) { const auto& L = a.L[l]; int U = 16, D = L.D, N1 = L.N1;
        std::vector<double> H((size_t)U * 4 * D), p((size_t)U * D), W((size_t)U * 16 * N1), M((size_t)U * N1), A((size_t)U * N1), m((size_t)U * 16);
        std::vector<double> Q((size_t)U * 4); L.get_hyperplanes_raw(H.data(), Q.data());
        for (int k = 0; k < U; k++) { for (int i = 0; i < D; i++) p[k * D + i] = L.gate_weight(k, i); for (int c = 0; c < 16; c++) { m[k * 16 + c] = L.competence(k, c); for (int j = 0; j < N1; j++) W[(k * 16 + c) * N1 + j] = L.rule(k, c, j); }
          for (int j = 0; j < N1; j++) { M[k * N1 + j] = L.active(k, j); A[k * N1 + j] = L.activity(k, j); } }
        b.L[l].import_dense(H.data(), p.data(), W.data(), M.data(), A.data(), m.data(), L.pruned, Q.data()); }
      b.t = a.t; std::vector<double> o1, o2; run_stream(a, 8, 800, 7, &nm, &o1); run_stream(b, 8, 800, 7, &nm, &o2);
      CHECK(o1 == o2, Nwarm < 512 ? "import_dense round trip before the prune continues identically" : "import_dense round trip after the prune continues identically"); } }
  printf(fails ? "%d test(s) FAILED\n" : "all tests passed\n", fails); return fails ? 1 : 0;
}
