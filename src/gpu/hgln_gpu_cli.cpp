// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_gpu - the HGLN kernel on a CUDA or HIP GPU through FORGE's runtime (include/hgln_gpu.hpp).
//   hgln_gpu info                                 drivers, devices and the image each would use (or why not)
//   hgln_gpu selftest d nets T [cuda|hip] [f32] [v42]   GPU vs CPU library: max |prediction diff| and |state diff|
//   hgln_gpu bench d nets T [cuda|hip] [f32] [v42]      time T samples for `nets` networks (all in parallel on the GPU)
// v42: the version 4.2 competence-weighted mixture (Params::v42); default the 4.3 plain average.
// Environment: HGLN_GPU_THREADS (threads per block, default 256), HGLN_GPU_SCRATCH (shared|global|automatic),
// HGLN_GPU_IMAGE (force an image, e.g. sm_70 or gfx1100). Exit 0 = ok, 1 = failure, 77 = no device.
// Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#include "hgln_gpu.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace hgln;
using hgln::gpu::Api;

static gpu::Options options() {
  gpu::Options o;
  if (const char* e = std::getenv("HGLN_GPU_THREADS")) o.threads = std::atoi(e);
  if (const char* e = std::getenv("HGLN_GPU_SCRATCH"))
    o.scratch = !std::strcmp(e, "shared") ? gpu::Scratch::shared : !std::strcmp(e, "global") ? gpu::Scratch::global : gpu::Scratch::automatic;
  return o;
}

static int info() {
  for (Api a : {Api::cuda, Api::hip}) {
    const int n = gpu::device_count(a);
    std::printf("%s: %d device(s)\n", gpu::api_name(a), n);
    for (int i = 0; i < std::max(n, 1); i++) {
      gpu::Bank<double> b; gpu::Bank<float> f; std::string w, wf;
      const bool ok = b.open(a, i, &w), okf = f.open(a, i, &wf);
      if (!ok && !okf) { std::printf("  [%d] unavailable: %s\n", i, w.c_str()); continue; }
      std::printf("  [%d] %s, max dynamic shared memory per block %zu bytes\n", i, (ok ? b.device_name() : f.device_name()).c_str(),
                  ok ? b.max_shared_bytes() : f.max_shared_bytes());
      std::printf("      float64 image: %s\n", ok ? b.image().c_str() : ("none (" + w + ")").c_str());
      std::printf("      float32 image: %s\n", okf ? f.image().c_str() : ("none (" + wf + ")").c_str());
    }
  }
  return 0;
}

template <class T> static int run(bool self, int d, int nets, int N, Api api, bool v42) {
  if (d < 1 || d > 128 || nets < 1 || N < 1) { std::fprintf(stderr, "need 1 <= d <= 128, nets >= 1, T >= 1\n"); return 1; }
  gpu::Bank<T> bank; std::string why;
  if (!bank.open(api, 0, &why, options())) { std::printf("no GPU: %s\n", why.c_str()); return 77; }
  Rng rng(12345);
  std::vector<T> X((size_t)N * d), Y((size_t)nets * N), P;
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < d; i++) X[(size_t)t * d + i] = (T)(2 * rng.uni() - 1);
    const T* x = &X[(size_t)t * d];
    T y = std::sin(3 * x[0]) * std::cos(2 * x[1 % d]) + (T)0.5 * x[2 % d] * x[3 % d];
    for (int n = 0; n < nets; n++) Y[(size_t)n * N + t] = n % 3 == 0 ? y : (n % 3 == 1 ? -y : (T)0.5 * y);
  }
  Params prm(d > 16 ? 0.02 : 0.2, d > 16 ? 0.02 : 0.05); prm.competence = v42;
  std::vector<Net<T>> nn(nets);
  for (int n = 0; n < nets; n++) nn[n].init(d, 7 + n, prm);
  if (!bank.upload(nn, &why)) { std::printf("upload failed: %s\n", why.c_str()); return 1; }
  std::printf("%s, image %s, %s, %s, scratch %zu bytes per network in %s memory\n", bank.device_name().c_str(), bank.image().c_str(),
              sizeof(T) == 8 ? "float64" : "float32", v42 ? "competence (4.2)" : "plain average (4.3)", bank.scratch_bytes(),
              bank.shared_scratch() ? "shared" : "global");
  auto t0 = std::chrono::steady_clock::now();
  if (!bank.run(X, Y, P, N, &why)) { std::printf("run failed: %s\n", why.c_str()); return 1; }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("GPU: %d networks x %d samples in %.2f ms = %.2f us per sample (all networks in parallel; includes copies)\n", nets, N, ms, 1000.0 * ms / N);
  if (!self) return 0;
  std::vector<Net<T>> got;
  if (!bank.download(got, &why)) { std::printf("download failed: %s\n", why.c_str()); return 1; }
  double wp = 0, ws = 0;
  for (int n = 0; n < nets; n++) {
    for (int t = 0; t < N; t++) { T pr = nn[n].step(&X[(size_t)t * d], Y[(size_t)n * N + t]); wp = std::max(wp, std::fabs((double)pr - (double)P[(size_t)n * N + t])); }
    for (int l = 0; l < 3; l++) {
      const auto &A = got[n].L[l], &B = nn[n].L[l];
      for (int k = 0; k < 16; k++) {
        for (int q = 0; q < 4; q++) {
          ws = std::max(ws, std::fabs((double)A.hyperplane_sqnorm(k, q) - (double)B.hyperplane_sqnorm(k, q)));
          for (int i = 0; i < A.D; i++) ws = std::max(ws, std::fabs((double)A.hyperplane_raw(k, q, i) - (double)B.hyperplane_raw(k, q, i)));
        }
        for (int i = 0; i < A.D; i++) ws = std::max(ws, std::fabs((double)A.gate_weight(k, i) - (double)B.gate_weight(k, i)));
        for (int c = 0; c < 16; c++) ws = std::max(ws, std::fabs((double)A.competence(k, c) - (double)B.competence(k, c)));
        for (int j = 0; j < A.N1; j++) {
          if (A.active(k, j) != B.active(k, j)) ws = INFINITY;
          else if (A.active(k, j)) for (int c = 0; c < 16; c++) ws = std::max(ws, std::fabs((double)A.rule(k, c, j) - (double)B.rule(k, c, j)));
        }
      }
    }
    if (got[n].t != nn[n].t) ws = INFINITY;
  }
  // on hardware the device exp/sqrt are not the C library's, so float64 agrees to rounding, not bit for bit
  const double tp = sizeof(T) == 8 ? 1e-9 : 1e-4, ts = sizeof(T) == 8 ? 1e-9 : 1e-3;
  std::printf("selftest %s d=%d nets=%d T=%d competence=%d: max |prediction diff| = %.3g, max |state diff| = %.3g (%s)\n", gpu::api_name(api), d, nets,
              N, v42 ? 1 : 0, wp, ws, sizeof(T) == 8 ? "float64" : "float32");
  return wp <= tp && ws <= ts ? 0 : 1;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "info";
  if (mode == "info") return info();
  if ((mode != "selftest" && mode != "bench") || argc < 5) {
    std::fprintf(stderr, "usage: hgln_gpu info | selftest d nets T [cuda|hip] [f32] [v42] | bench d nets T [cuda|hip] [f32] [v42]\n");
    return 1;
  }
  const int d = std::atoi(argv[2]), nets = std::atoi(argv[3]), N = std::atoi(argv[4]);
  Api api = Api::cuda; bool f32 = false, v42 = false;
  for (int i = 5; i < argc; i++) {
    if (!std::strcmp(argv[i], "hip")) api = Api::hip;
    if (!std::strcmp(argv[i], "f32")) f32 = true;
    if (!std::strcmp(argv[i], "v42")) v42 = true;
  }
  const bool self = mode == "selftest";
  if (f32) return run<float>(self, d, nets, N, api, v42);
  int r = run<double>(self, d, nets, N, api, v42);
  if (self && r == 0) r = run<float>(self, d, nets, N, api, v42);   // selftest checks both real types
  return r;
}
