// SPDX-License-Identifier: AGPL-3.0-or-later
// test_gpu.cpp - hgln::gpu::Bank end to end: Net -> Bank -> FORGE gpu::cuda_hip runtime -> driver -> hgln_run ->
// results, compared with the CPU library sample by sample. Under ctest the driver is HGLN's mock (tests/mock_hgln.c),
// which runs the real kernel source on the CPU, so float64 must agree EXACTLY (0 difference) and float32 to about
// 1e-6. On a machine with a real GPU and no mock on the library path the same program tests the hardware (then
// float64 can differ in the last bits: the device exp and sqrt are not the C library's).
//   hgln_test_gpu cuda|hip       all checks on that API (d = 8, 64, 100; float64 and float32; shared and global
//                                scratch; several calls with state resident; download, re-upload, continue)
//   hgln_test_gpu nodriver       open() without a driver must fail with a reason (exit 77 if a driver is present)
// Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#include "hgln_gpu.hpp"
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace hgln;
using hgln::gpu::Api;

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); failures++; } } while (0)

// The mock driver's view of the last hgln_run launch (nullptr on a real driver).
using last_fn = void (*)(unsigned*, unsigned*, unsigned*, unsigned*, unsigned*, char*);
static last_fn mock_last(Api api) {
  void* h = dlopen(api == Api::cuda ? "libcuda.so.1" : "libamdhip64.so.6", RTLD_NOW | RTLD_NOLOAD);
  return h ? reinterpret_cast<last_fn>(dlsym(h, "mock_hgln_last_launch")) : nullptr;
}

template <class T> static double state_diff(const Net<T>& a, const Net<T>& b) {
  double mx = 0;
  auto upd = [&](double x, double y) { mx = std::max(mx, std::fabs(x - y)); };
  if (a.t != b.t || a.d != b.d) return INFINITY;
  for (int l = 0; l < 3; l++) {
    const auto &A = a.L[l], &B = b.L[l];
    if (A.pruned != B.pruned) return INFINITY;
    for (int k = 0; k < 16; k++) {
      for (int q = 0; q < 4; q++) for (int i = 0; i < A.D; i++) upd(A.hyperplane(k, q, i), B.hyperplane(k, q, i));
      for (int i = 0; i < A.D; i++) upd(A.gate_weight(k, i), B.gate_weight(k, i));
      for (int c = 0; c < 16; c++) upd(A.competence(k, c), B.competence(k, c));
      for (int j = 0; j < A.N1; j++) {
        if (A.active(k, j) != B.active(k, j)) return INFINITY;
        if (A.active(k, j)) for (int c = 0; c < 16; c++) upd(A.rule(k, c, j), B.rule(k, c, j));
        if (!A.pruned) upd(A.activity(k, j), B.activity(k, j));
      }
    }
  }
  return mx;
}

template <class T> struct Stream {
  int d, nets, N; std::vector<T> X, Y;   // X: N x d; Y: nets x N
  Stream(int d_, int nets_, int N_, uint64_t seed) : d(d_), nets(nets_), N(N_), X((size_t)N_ * d_), Y((size_t)nets_ * N_) {
    Rng rng(seed);
    for (int t = 0; t < N; t++) {
      for (int i = 0; i < d; i++) X[(size_t)t * d + i] = (T)(2 * rng.uni() - 1);
      const T* x = &X[(size_t)t * d];
      T y = std::sin(3 * x[0]) * std::cos(2 * x[1 % d]) + (T)0.5 * x[2 % d] * x[3 % d];
      for (int n = 0; n < nets; n++) Y[(size_t)n * N + t] = n % 3 == 0 ? y : (n % 3 == 1 ? -y : (T)0.5 * y);
    }
  }
  // samples [s, s + len): X rows and per-network Y rows, as Bank::run takes them
  void chunk(int s, int len, std::vector<T>& x, std::vector<T>& y) const {
    x.assign(X.begin() + (size_t)s * d, X.begin() + (size_t)(s + len) * d);
    y.resize((size_t)nets * len);
    for (int n = 0; n < nets; n++) for (int i = 0; i < len; i++) y[(size_t)n * len + i] = Y[(size_t)n * N + s + i];
  }
};

// CPU reference for one chunk; returns max |P - reference prediction|
template <class T> static double ref_chunk(std::vector<Net<T>>& ref, const std::vector<T>& x, const std::vector<T>& y, const std::vector<T>& P, int len) {
  double mx = 0; const int d = ref[0].d, nets = (int)ref.size();
  for (int i = 0; i < len; i++)
    for (int n = 0; n < nets; n++) {
      T pr = ref[n].predict(&x[(size_t)i * d]); ref[n].learn(y[(size_t)n * len + i]);
      mx = std::max(mx, std::fabs((double)pr - (double)P[(size_t)n * len + i]));
    }
  return mx;
}

template <class T> static void one(Api api, int d, int nets, int N, gpu::Scratch scratch, int threads, const std::vector<int>& chunks) {
  const bool f64 = sizeof(T) == 8;
  const double tolP = f64 ? 0 : 1e-5, tolS = f64 ? 0 : 1e-4;
  Params prm(d > 16 ? 0.02 : 0.2, d > 16 ? 0.02 : 0.05);
  std::vector<Net<T>> init(nets);
  for (int n = 0; n < nets; n++) init[n].init(d, 7 + n, prm);
  std::vector<Net<T>> ref = init;
  Stream<T> st(d, nets, N, 12345 + d);

  gpu::Bank<T> bank; std::string why;
  gpu::Options opt; opt.threads = threads; opt.scratch = scratch;
  if (!bank.open(api, 0, &why, opt)) { CHECK(false, "open: %s", why.c_str()); return; }
  if (!bank.upload(init, &why)) {
    if (scratch == gpu::Scratch::shared && why.find("shared memory") != std::string::npos) {   // a small device: refused, as it must be
      std::printf("%s %s d=%d: shared scratch refused: %s\n", gpu::api_name(api), f64 ? "float64" : "float32", d, why.c_str());
      return;
    }
    CHECK(false, "upload: %s", why.c_str()); return;
  }
  CHECK(bank.size() == (size_t)nets && bank.d() == d, "bank size/d");
  if (scratch == gpu::Scratch::shared) CHECK(bank.shared_scratch(), "shared scratch requested but not used");
  if (scratch == gpu::Scratch::global) CHECK(!bank.shared_scratch(), "global scratch requested but shared used");
  if (scratch == gpu::Scratch::automatic)
    CHECK(bank.shared_scratch() == (bank.scratch_bytes() <= bank.max_shared_bytes()), "automatic scratch choice (%zu bytes, device %zu)",
          bank.scratch_bytes(), bank.max_shared_bytes());

  double wP = 0, wMid = 0; int s = 0, calls = 0;
  std::vector<T> x, y, P;
  last_fn last = mock_last(api);
  for (size_t c = 0; c < chunks.size() && s < N; c++) {
    const int len = std::min(chunks[c], N - s);
    st.chunk(s, len, x, y);
    if (!bank.run(x, y, P, len, &why)) { CHECK(false, "run: %s", why.c_str()); return; }
    calls++;
    if (last) {   // what the driver saw
      unsigned L = 0, g = 0, b = 0, sm = 0, real = 0; char arch[16] = {};
      last(&L, &g, &b, &sm, &real, arch);
      CHECK(g == (unsigned)nets && b == (unsigned)threads && real == sizeof(T), "launch grid %u block %u real %u", g, b, real);
      CHECK(sm == (bank.shared_scratch() ? bank.scratch_bytes() : 0), "launch smem %u", sm);
    }
    wP = std::max(wP, ref_chunk(ref, x, y, P, len));
    s += len;
    CHECK(bank.samples_seen(0) == s, "samples_seen %ld != %d", bank.samples_seen(0), s);
    if (c == 1) {   // download mid-stream: must match and must not disturb the device state
      std::vector<Net<T>> mid;
      if (!bank.download(mid, &why)) { CHECK(false, "download: %s", why.c_str()); return; }
      for (int n = 0; n < nets; n++) wMid = std::max(wMid, state_diff(mid[n], ref[n]));
    }
  }
  std::vector<Net<T>> got = init;   // overwrite existing networks
  if (!bank.download(got, &why)) { CHECK(false, "download: %s", why.c_str()); return; }
  double wS = 0; for (int n = 0; n < nets; n++) wS = std::max(wS, state_diff(got[n], ref[n]));
  // round trip: upload the downloaded networks (now past the prune) and continue on the device and on the CPU
  double wR = 0;
  {
    const int extra = 64; Stream<T> st2(d, nets, extra, 999 + d);
    if (!bank.upload(got, &why)) { CHECK(false, "re-upload: %s", why.c_str()); return; }
    st2.chunk(0, extra, x, y);
    if (!bank.run(x, y, P, extra, &why)) { CHECK(false, "run after re-upload: %s", why.c_str()); return; }
    wR = ref_chunk(ref, x, y, P, extra);
    std::vector<Net<T>> fin;
    if (!bank.download(fin, &why)) { CHECK(false, "download: %s", why.c_str()); return; }
    for (int n = 0; n < nets; n++) wR = std::max(wR, state_diff(fin[n], ref[n]));
  }
  std::printf("%s %s d=%d nets=%d T=%d calls=%d threads=%d scratch=%s(%zu B) image=%s: max |P diff| = %.3g, state diff mid = %.3g, end = %.3g, after re-upload = %.3g\n",
              gpu::api_name(api), f64 ? "float64" : "float32", d, nets, N, calls, threads, bank.shared_scratch() ? "shared" : "global",
              bank.scratch_bytes(), bank.image().c_str(), wP, wMid, wS, wR);
  CHECK(wP <= tolP && wMid <= tolS && wS <= tolS && wR <= std::max(tolP, tolS), "differences above tolerance (P %.3g, state %.3g)", tolP, tolS);
  std::fflush(stdout);
}

static int nodriver() {
  int present = 0;
  for (Api a : {Api::cuda, Api::hip}) {
    gpu::Bank<double> b; std::string why;
    if (b.open(a, 0, &why)) { std::printf("%s: a driver is present (%s): skip\n", gpu::api_name(a), b.device_name().c_str()); present++; continue; }
    std::printf("%s: open failed as expected: %s\n", gpu::api_name(a), why.c_str());
    CHECK(!why.empty(), "no reason given");
    CHECK(gpu::device_count(a) == 0, "device_count without a driver");
  }
  if (failures) return 1;
  return present == 2 ? 77 : 0;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "cuda";
  if (mode == "nodriver") return nodriver();
  const Api api = mode == "hip" ? Api::hip : Api::cuda;
  const int n = gpu::device_count(api);
  std::printf("%s: %d device(s)\n", gpu::api_name(api), n);
  if (n < 1) { std::printf("skip: no %s driver/device\n", gpu::api_name(api)); return 77; }
  {   // device and image selection (as the mock is configured by MOCK_ARCH)
    gpu::Bank<double> b; std::string why;
    CHECK(b.open(api, 0, &why), "open: %s", why.c_str());
    std::printf("device: %s, float64 image: %s, max shared/block: %zu\n", b.device_name().c_str(), b.image().c_str(), b.max_shared_bytes());
    if (mock_last(api)) {
      const char* ma = std::getenv("MOCK_ARCH");
      std::string want = api == Api::cuda ? (ma && std::string(ma) == "sm_70" ? "ptx sm_70" : "ptx sm_80") : (ma && *ma ? ma : "gfx90a");
      CHECK(b.image() == want, "image %s, expected %s", b.image().c_str(), want.c_str());
    }
    gpu::Bank<double> bad; std::string w2;
    CHECK(!bad.open(api, 5, &w2) && !w2.empty(), "ordinal out of range must fail");
    std::printf("ordinal 5: %s\n", w2.c_str());
    gpu::Options o; o.threads = 2000;
    CHECK(!bad.open(api, 0, &w2, o), "threads > 1024 must fail");
  }
  using S = gpu::Scratch;
  const std::vector<int> chunks = {1, 200, 311, 3, 185};   // crosses the prune at sample 512
  for (S sc : {S::shared, S::global}) {
    one<double>(api, 8, 3, 700, sc, 32, chunks);
    one<double>(api, 64, 2, 600, sc, 64, chunks);
    one<double>(api, 100, 2, 560, sc, 64, chunks);
    one<float>(api, 8, 3, 700, sc, 32, chunks);
    one<float>(api, 64, 2, 600, sc, 64, chunks);
    one<float>(api, 100, 2, 560, sc, 64, chunks);
  }
  one<double>(api, 64, 2, 300, S::automatic, 256, {120, 180});   // the default block size, automatic placement
  one<float>(api, 8, 4, 300, S::automatic, 256, {300});
  if (failures) { std::printf("%d FAILURES\n", failures); return 1; }
  std::printf("ALL GPU TESTS PASSED (%s)\n", gpu::api_name(api));
  return 0;
}
