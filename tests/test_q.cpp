// SPDX-License-Identifier: AGPL-3.0-or-later
// test_q.cpp - the integer net (hgln::NetQ, include/hgln_q.hpp) against the research reference of the v5.0 integer
// specification (research/alg/hgln_var.hpp, QNet / intspec=1): every prediction and the whole model state must be
// bit-identical, for d = 8 and 64, 16/12/8 units, both bit widths, the v5 defaults and the 4.3-like configuration,
// past the prune and beyond 4096 samples. Also: Philox known-answer vectors, the sigmoid table, the integer helpers,
// infer/run_streams/Classifier consistency. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#include "hgln_q.hpp"
#include "../research/alg/hgln_var.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace hgln;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static void unit_tests() {
  // Philox4x32-10 known-answer vectors (Random123 kat_vectors)
  struct K { uint32_t c[4], k[2], r[4]; } kat[3] = {
      {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
      {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu}, {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
      {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u}, {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}}};
  for (auto& v : kat) {
    const qspec::Philox4 o = qspec::philox4x32_10(v.c[0], v.c[1], v.c[2], v.c[3], v.k[0], v.k[1]);
    CHECK(!std::memcmp(o.v, v.r, 16), "Philox4x32-10 known-answer vector");
  }
  // the lane-parallel Philox of the kernels equals the scalar one
  uint32_t out[4][16];
  detail::qk::philox_lanes<detail::qk::HeaderPolicy, 16>(0x12340, 7, 99, 1, 0xdeadbeefu, 42, out);
  bool lanes_ok = true;
  for (int l = 0; l < 16; l++) { const qspec::Philox4 o = qspec::philox4x32_10(0x12340 + l, 7, 99, 1, 0xdeadbeefu, 42); for (int w = 0; w < 4; w++) lanes_ok &= o.v[w] == out[w][l]; }
  CHECK(lanes_ok, "lane-parallel Philox equals the scalar Philox");
  // sigmoid table: within 1 of the exact values, symmetric, monotone
  int bad = 0;
  for (int i = 0; i <= 1024; i++) {
    double z = (i - 512) / 512.0, sg = std::min(std::max(1 / (1 + std::exp(-12 * z)), 1e-4), 1 - 1e-4);
    bad += std::fabs(qspec::kSigLut[i] - 16384 * sg) > 0.5001 || qspec::kSigLut[i] + qspec::kSigLut[1024 - i] != 16384 || (i && qspec::kSigLut[i] < qspec::kSigLut[i - 1]);
  }
  CHECK(bad == 0, "sigmoid table: %d bad entries", bad);
  int sbad = 0;
  for (int v = -20000; v <= 20000; v += 7) {
    const double ex = 16384 * std::min(std::max(1 / (1 + std::exp(-12.0 * v / 16384)), 1e-4), 1 - 1e-4);
    sbad += std::fabs(qspec::sig14(v) - ex) > 1.02;
  }
  CHECK(sbad == 0, "interpolated sigmoid within 1.02 LSB of Q14 (%d off)", sbad);
  // helpers
  CHECK(qspec::rne(5, 1) == 2 && qspec::rne(7, 1) == 4 && qspec::rne(-5, 1) == -2 && qspec::rne(-7, 1) == -4 && qspec::rne(6, 2) == 2 && qspec::rne(-6, 2) == -2, "rne ties to even");
  CHECK(qspec::div_rne(5, 2) == 2 && qspec::div_rne(7, 2) == 4 && qspec::div_rne(-5, 2) == -2 && qspec::div_rne(-7, 3) == -2 && qspec::div_rne(10, 4) == 2, "div ties to even");
  bool sq = true;
  for (uint64_t n : {0ull, 1ull, 2ull, 3ull, 4ull, 15ull, 16ull, 17ull, 1ull << 40, (1ull << 40) - 1, 0xFFFFFFFFFFFFull, ~0ull >> 1}) { uint64_t r = qspec::isqrt(n); sq &= r * r <= n && (r + 1) * (r + 1) > n; }
  CHECK(sq, "isqrt exact");
  CHECK(qspec::sr(5 << 16, 16, 0) == 5 && qspec::sr((5 << 16) + 0x8000, 16, 0x7FFF) == 6 && qspec::sr((5 << 16) + 0x8000, 16, 0x8000) == 5 && qspec::sr(-1, 16, 0xFFFE) == 0, "stochastic rounding");
  const int32_t g0 = qspec::gl_q24(0, 0.2, 0.05, 1000), g1 = qspec::gl_q24(1000, 0.2, 0.05, 1000), g2 = qspec::gl_q24(100000, 0.2, 0.05, 1000);
  CHECK(std::abs(g0 - 3355443) <= 1 && std::abs(g1 - (int)std::nearbyint((0.05 + 0.15 * std::exp(-1.0)) * 16777216)) <= 2 && g2 == 838861, "gate-rate schedule (%d %d %d)", g0, g1, g2);
}

// research reference configured like Params
static hv::Var var_of(const Params& P) {
  hv::Var V; V.uniform = true; V.U = P.units; V.hybrid = P.fanout ? 3 : 0; V.glob = P.glob; V.mu = P.mu; V.glr0 = P.glr0; V.glr1 = P.glr1;
  V.tau = P.tau; V.warm = P.warm; V.theta = P.latch; V.theta_mode = 4; V.credit = P.latch_credit > 0; V.credit_scale = P.latch_credit;
  V.gate_err = P.gate_err; V.sparse_k = P.corners; V.wbits = P.wbits; V.abits = P.abits; V.intspec = true; return V;
}
template <class R> static bool same_state(const NetQ<R>& a, const hv::QNet& r, std::string& why) {
  if (a.t != r.t || a.pruned() != r.pruned) { why = "t/pruned"; return false; }
  for (int l = 0; l < 3; l++) {
    const LayerQ& L = a.L[l]; const hv::QNet::QL& Q = r.L[l];
    for (int k = 0; k < L.U; k++) {
      for (int i = 0; i < L.D; i++) {
        if (L.gate_weight(k, i) != Q.p[(size_t)k * L.D + i]) { why = "p"; return false; }
        for (int b = 0; b < 4; b++) if (L.hyperplane(k, b, i) != Q.H[((size_t)k * 4 + b) * L.D + i]) { why = "H"; return false; }
      }
      for (int j = 0; j < L.N1; j++) {
        if (L.active(k, j) != (Q.M[(size_t)k * L.N1 + j] != 0)) { why = "M"; return false; }
        if (!L.pruned && L.activity(k, j) != Q.A[(size_t)k * L.N1 + j]) { why = "A"; return false; }
        for (int c = 0; c < 16; c++) if (L.rule(k, c, j) != (Q.M[(size_t)k * L.N1 + j] ? Q.W[((size_t)k * 16 + c) * L.N1 + j] : 0)) { why = "W"; return false; }
      }
    }
  }
  return true;
}
// One stream through both implementations; inputs include large values (saturation) and +-1 targets.
static void compare(const char* name, int d, int N, uint64_t seed, const Params& P) {
  NetQ<double> a; a.init(d, seed, P);
  hv::QNet r; r.init(d, seed, var_of(P));
  std::string why;
  CHECK(same_state(a, r, why), "%s: initial state differs (%s)", name, why.c_str());
  Rng rng(seed * 31 + 7); std::vector<double> x(d);
  int first_bad = -1; double maxdiff = 0;
  for (int t = 0; t < N; t++) {
    const double sc = (t % 97 == 0) ? 40.0 : 1.0;
    for (int i = 0; i < d; i++) x[i] = sc * (2 * rng.uni() - 1);
    double y = std::sin(3 * x[0]) * std::cos(2 * x[1 % d]) + 0.5 * x[2 % d] * x[3 % d];
    if (t % 5 == 0) y = y > 0 ? 1 : -1;
    if (t % 211 == 0) y = 100;
    const double pa = a.predict(x.data()), pr = r.predict(x.data());
    if (pa != pr && first_bad < 0) first_bad = t;
    maxdiff = std::max(maxdiff, std::fabs(pa - pr));
    a.learn(y); r.learn(y);
  }
  const bool st = same_state(a, r, why);
  CHECK(first_bad < 0 && st, "%s: library differs from the reference (first prediction %d, max diff %g, state %s)", name, first_bad, maxdiff, st ? "same" : why.c_str());
  std::printf("  %-34s d=%-3d U=%-2d bits %d/%d corners %-2d N=%d: %s\n", name, d, P.units, P.wbits, P.abits, P.corners, N,
              first_bad < 0 && st ? "bit-identical (predictions and state)" : "DIFFERENT");
}

int main() {
  unit_tests();
  std::printf("library vs research reference (intspec):\n");
  for (int bits : {16, 12})
    for (int units : {8, 12, 16})
      for (int d : {8, 64}) {
        Params P = d > 16 ? Params::v5(0.02, 0.02) : Params::v5(); P.units = units; P.wbits = P.abits = bits;
        compare("v5 defaults", d, d == 8 ? 4600 : 4200, 11 + d + units, P);
      }
  {  // 4.3-like: chain, plain gates, mixed gate error, all 16 corners
    Params P; P.wbits = 16; P.abits = 14; compare("4.3 structure (chain, dense)", 8, 4400, 5, P);
    P.units = 12; P.corners = 8; P.wbits = 13; P.abits = 16; P.gate_err = 1; compare("chain, top-8, own error", 64, 1300, 6, P);
    Params Q = Params::v5(); Q.fanout = false; Q.latch_credit = 1.0; Q.corners = 1; compare("chain latch, credit 1, top-1", 8, 2000, 7, Q);
  }
  {  // float API == double API on float inputs; infer == predict; run_streams; Classifier
    NetQ<float> f; NetQ<double> g; f.init(8, 3); g.init(8, 3);
    Rng rng(4); std::vector<float> xf(8); std::vector<double> xd(8); bool same = true;
    for (int t = 0; t < 1500; t++) {
      for (int i = 0; i < 8; i++) { xf[i] = (float)(2 * rng.uni() - 1); xd[i] = xf[i]; }
      const float yf = (float)std::sin(3.0 * xf[0]);
      same &= (float)g.predict(xd.data()) == f.predict(xf.data());
      f.learn(yf); g.learn((double)yf);
    }
    CHECK(same && f.ws.S == g.ws.S, "NetQ<float> follows NetQ<double> exactly on float inputs");
    WorkQ w; w.init(f); bool inf = true;
    for (int t = 0; t < 50; t++) { for (auto& v : xf) v = (float)(2 * rng.uni() - 1); inf &= f.infer(xf.data(), w) == f.predict(xf.data()); }
    CHECK(inf, "infer with a separate Work equals predict");
    const size_t N = 700; std::vector<float> X(N * 8), Y(2 * N), P1(2 * N), P2(2 * N);
    for (auto& v : X) v = (float)(2 * rng.uni() - 1);
    for (size_t t = 0; t < N; t++) { Y[t] = std::sin(3 * X[t * 8]); Y[N + t] = X[t * 8 + 1] * X[t * 8 + 2]; }
    std::vector<NetQ<float>> A(2), B(2);
    for (int k = 0; k < 2; k++) { A[k].init(8, 10 + k); B[k].init(8, 10 + k); }
    run_streams(A, X.data(), Y.data(), P1.data(), N, 2);
    for (int k = 0; k < 2; k++) for (size_t t = 0; t < N; t++) { P2[k * N + t] = B[k].predict(&X[t * 8]); B[k].learn(Y[k * N + t]); }
    CHECK(P1 == P2, "run_streams equals the sequential loop");
    run_streams(A, X.data(), (const float*)nullptr, P1.data(), N, 3);
    for (int k = 0; k < 2; k++) for (size_t t = 0; t < N; t++) P2[k * N + t] = B[k].predict(&X[t * 8]);
    CHECK(P1 == P2, "run_streams predict-only equals predict");
    Classifier<float, NetQ<float>> c1, c2; c1.init(8, 3, 1); c2.init(8, 3, 1);
    std::vector<int> lab(N); for (size_t t = 0; t < N; t++) lab[t] = (int)(t % 3);
    c1.train_batch(X.data(), lab.data(), N, 3);
    for (size_t t = 0; t < N; t++) c2.train(&X[t * 8], lab[t]);
    std::vector<int> o1(N); c1.predict_batch(X.data(), N, o1.data(), nullptr, 2); bool cs = true;
    for (size_t t = 0; t < N; t++) cs &= o1[t] == c2.predict(&X[t * 8]);
    CHECK(cs, "Classifier<float, NetQ<float>>: train_batch == train, predict_batch == predict");
    bool threw = false; try { NetQ<float> q; Params P = Params::v5(); P.wbits = 11; q.init(4, 1, P); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw, "bit widths outside 12..16 rejected");
    std::printf("  model bytes per net (v5, d=8 / d=64, after the prune): %zu / ", f.model_bytes());
    NetQ<float> big; big.init(64, 1, Params::v5(0.02, 0.02));
    std::vector<float> xb(64); for (int t = 0; t < 600; t++) { for (auto& v : xb) v = (float)(2 * rng.uni() - 1); big.predict(xb.data()); big.learn(xb[0]); }
    std::printf("%zu\n", big.model_bytes());
  }
  if (fails) { std::printf("%d FAILURES\n", fails); return 1; }
  std::printf("all integer-net tests passed\n");
  return 0;
}
