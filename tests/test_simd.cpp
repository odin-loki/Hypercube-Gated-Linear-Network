// SPDX-License-Identifier: AGPL-3.0-or-later
// test_simd.cpp - every SIMD kernel table usable on this machine gives bit-identical predictions and model state,
// equal to the header-only kernels (Layer::forward_k/update_k with the vector-extension policy, compiled here).
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
#include "hgln.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace hgln;

namespace {
// The header-only kernels as a table, so the same Net code can be run through them.
struct RefF : detail::VxOps<float> {};
struct RefD : detail::VxOps<double> {};
template <class T, class P> void rfwd(const Layer<T>& L, const T* in, const T* u, LayerWork<T>& w, T* xg) { L.template forward_k<P>(in, u, w, xg); }
template <class T, class P> void rupd(Layer<T>& L, T y, T mu, T glr, const T* es, const T* u, const LayerWork<T>& w) { L.template update_k<P>(y, mu, glr, es, u, w); }
const simd::Table header_table = {"header", 0, {&rfwd<float, RefF>, &rupd<float, RefF>}, {&rfwd<double, RefD>, &rupd<double, RefD>}};

int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

template <class T> void put(std::vector<unsigned char>& s, const T* p, size_t n) {
  const unsigned char* b = reinterpret_cast<const unsigned char*>(p); s.insert(s.end(), b, b + n * sizeof(T));
}
template <class T> std::vector<unsigned char> state(const Net<T>& g) {      // the whole model, byte for byte
  std::vector<unsigned char> s;
  put(s, &g.t, 1);
  for (const auto& l : g.L) {
    int hdr[3] = {l.cs, (int)l.pruned, l.nt}; put(s, hdr, 3);
    put(s, l.H.data(), l.H.size()); put(s, l.p.data(), l.p.size()); put(s, l.W.data(), l.W.size());
    put(s, l.m.data(), l.m.size()); put(s, l.A.data(), l.A.size());
    put(s, l.cols.data(), l.cols.size()); put(s, l.ncol.data(), l.ncol.size());
  }
  return s;
}
struct Run { std::vector<unsigned char> pred, outs, model, infer; };

// One stream: learn N samples (predictions and the 48 unit outputs recorded every step), then predict-only through
// infer() and run_streams() on a second block. Inputs include values large enough to saturate the gates.
template <class T> Run run(int d, int N, uint64_t seed) {
  Net<T> g; Params P; if (d >= 32) P = Params(0.02, 0.02);
  g.init(d, seed, P);
  Rng rng(seed * 7919 + 13);
  std::vector<T> x(d);
  Run r;
  for (int t = 0; t < N; t++) {
    const double sc = (t % 97 == 0) ? 40.0 : 1.0;
    for (int i = 0; i < d; i++) x[i] = (T)(sc * (2 * rng.uni() - 1));
    T y = (T)(std::sin(3 * (double)x[0]) * std::cos(2 * (double)x[1 % d]) + 0.5 * (double)x[2 % d] * (double)x[3 % d]);
    if (t % 5 == 0) y = (T)(y > 0 ? 1 : -1);
    T p = g.predict(x.data()); g.learn(y);
    put(r.pred, &p, 1); put(r.outs, g.outputs(), 48);
  }
  r.model = state(g);
  const int M = 200; std::vector<T> X((size_t)M * d), out((size_t)2 * M);
  for (auto& v : X) v = (T)(2 * rng.uni() - 1);
  Work<T> w; w.init(g);
  for (int t = 0; t < M; t++) out[t] = g.infer(X.data() + (size_t)t * d, w);
  std::vector<Net<T>> nets(1, g);
  run_streams(nets, X.data(), (const T*)nullptr, out.data() + M, (size_t)M, 2);
  put(r.infer, out.data(), out.size());
  return r;
}

template <class T> void compare_all(int d, int N, uint64_t seed, const std::vector<const simd::Table*>& tabs) {
  simd::install(&header_table);
  const Run ref = run<T>(d, N, seed);
  for (const simd::Table* t : tabs) {
    simd::install(t);
    CHECK(&simd::active() == t, "install(%s)", t->target);
    const Run got = run<T>(d, N, seed);
    const bool ok = got.pred == ref.pred && got.outs == ref.outs && got.model == ref.model && got.infer == ref.infer;
    CHECK(ok, "%s d=%d %s: differs from the header-only kernels (pred %d outs %d model %d infer %d)", sizeof(T) == 4 ? "float32" : "float64",
          d, t->target, got.pred == ref.pred, got.outs == ref.outs, got.model == ref.model, got.infer == ref.infer);
    std::printf("  %-7s %s d=%-3d N=%d: %s (%zu prediction bytes, %zu model bytes)\n", t->target, sizeof(T) == 4 ? "float32" : "float64", d, N,
                ok ? "bit-identical" : "DIFFERENT", ref.pred.size() + ref.outs.size() + ref.infer.size(), ref.model.size());
  }
}
}  // namespace

int main() {
  const simd::Table& chosen = simd::active();
  std::printf("cpu: %s\nautomatic table: %s\n", simd::cpu_summary().c_str(), chosen.target);
  std::vector<const simd::Table*> tabs;
  for (const char* const* n = simd::targets(); *n; n++) {
    const simd::Table* t = simd::table(*n);
    std::printf("table %-7s %s\n", *n, t ? "usable" : "built, not usable on this CPU (skipped)");
    if (t) tabs.push_back(t);
  }
  CHECK(simd::table("scalar") != nullptr, "scalar table missing");
  CHECK(simd::table("no-such-target") == nullptr && !simd::force("no-such-target"), "unknown target accepted");
  CHECK(&simd::active() == &chosen, "a failed force() changed the table");
  for (int d : {3, 8, 64}) {
    const int N = d == 64 ? 1200 : 2500;     // past the prune at 512 and several structure steps
    compare_all<float>(d, N, 11 + d, tabs);
    compare_all<double>(d, N, 11 + d, tabs);
  }
  for (const simd::Table* t : tabs) { CHECK(simd::force(t->target) && &simd::active() == t, "force(%s)", t->target); }
  simd::install(nullptr);
  CHECK(&simd::active() == &chosen, "install(nullptr) did not return to the automatic choice");
  if (fails) { std::printf("%d FAILURES\n", fails); return 1; }
  std::printf("all %zu tables bit-identical to the header-only kernels\n", tabs.size());
  return 0;
}
