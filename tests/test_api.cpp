// test_api.cpp - behavioural tests for the header-only library. Author: Odin Loch
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
    int pruned = 0, total = 0; for (int l = 0; l < 3; l++) { for (int k = 0; k < 16; k++) { int act = 0; for (int j = 0; j < a.L[l].n; j++) act += a.L[l].M[k * a.L[l].N1 + j] > 0; total++; int want = (int)std::nearbyint(0.6 * a.L[l].n); pruned += act == want; if (a.L[l].M[k * a.L[l].N1 + a.L[l].n] != 1) pruned -= 1000; } }
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
  printf(fails ? "%d test(s) FAILED\n" : "all tests passed\n", fails); return fails ? 1 : 0;
}
