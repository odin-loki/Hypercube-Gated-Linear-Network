// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_gpu - the HGLN kernel on a CUDA or HIP GPU through FORGE's runtime (include/hgln_gpu.hpp).
//   hgln_gpu info                                 drivers, devices and the image each would use (or why not)
//   hgln_gpu selftest d nets T [cuda|hip] [f32] [v42] [units=8|12|16] [fanout] [v5] [int [v43]]   GPU vs CPU library: max |prediction
//                                                 diff| and |state diff|
//   hgln_gpu bench d nets T [cuda|hip] [f32] [v42] [units=N] [fanout] [v5] [int [v43]]   time T samples for `nets` networks (all in
//                                                 parallel on the GPU)
// Default: version 4.3 (Params::v43). v42: the version 4.2 competence-weighted mixture (Params::v42). units=N:
// Params::units (default 16). fanout: Params::fanout (layer 3 reads layer 1's outputs). v5: the version 5.0 float
// options of Params::v5() - cross-layer latch 0.1, latch credit 0.5, gates on the global error, glob 0.95, top-6 corners - with the
// given units / fanout (v5 itself uses units=8 fanout). int: the integer network hgln::NetQ (Params::v5(), or the 4.3
// structure with v43; units default 8 with int, 16 with int v43) through hgln::gpu::BankQ; the selftest then requires bit-identical predictions and state.
// Environment: HGLN_GPU_THREADS (threads per block, default automatic), HGLN_GPU_SCRATCH (shared|global|automatic),
// HGLN_GPU_KERNEL (block|warp|automatic: the float kernel),
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
  if (const char* e = std::getenv("HGLN_GPU_KERNEL"))
    o.kernel = !std::strcmp(e, "block") ? gpu::Kernel::block : !std::strcmp(e, "warp") ? gpu::Kernel::warp : gpu::Kernel::automatic;
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

template <class T> static int run(bool self, int d, int nets, int N, Api api, bool v42, int units, bool fanout, bool v5) {
  if (d < 1 || d > 128 || nets < 1 || N < 1 || !valid_units(units)) {
    std::fprintf(stderr, "need 1 <= d <= 128, nets >= 1, T >= 1, units 8, 12 or 16\n"); return 1;
  }
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
  Params prm = Params::v43(d > 16 ? 0.02 : 0.2, d > 16 ? 0.02 : 0.05); prm.competence = v42; prm.units = units; prm.fanout = fanout;
  if (v5) { prm.latch = 0.1; prm.latch_credit = 0.5; prm.gate_err = 2; prm.glob = 0.95; prm.corners = 6; }
  std::vector<Net<T>> nn(nets);
  for (int n = 0; n < nets; n++) nn[n].init(d, 7 + n, prm);
  if (!bank.upload(nn, &why)) { std::printf("upload failed: %s\n", why.c_str()); return 1; }
  std::printf("%s, image %s, %s kernel (%d per block, %d threads), %s, %s, %d units, %s%s, scratch %zu bytes per network in %s memory\n", bank.device_name().c_str(), bank.image().c_str(),
              bank.warp_kernel() ? "warp" : "block", bank.nets_per_block(), bank.threads_per_block(), sizeof(T) == 8 ? "float64" : "float32", v42 ? "competence (4.2)" : "plain average (4.3)", units, fanout ? "fanout" : "chain", v5 ? ", v5 latch/credit/global-error gates" : "", bank.scratch_bytes(),
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
      if (A.U != B.U) ws = INFINITY;
      for (int k = 0; k < std::min(A.U, B.U); k++) {
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
  // the kernel and the library do the same IEEE operations (no libm: hgln_gate, hgln_exp; sqrt and division are
  // correctly rounded on every device), so the GPU must reproduce the CPU bit for bit, float64 and float32
  const double tp = 0, ts = 0;
  std::printf("selftest %s d=%d nets=%d T=%d competence=%d units=%d %s%s: max |prediction diff| = %.3g, max |state diff| = %.3g (%s)\n", gpu::api_name(api),
              d, nets, N, v42 ? 1 : 0, units, fanout ? "fanout" : "chain", v5 ? " v5" : "", wp, ws, sizeof(T) == 8 ? "float64" : "float32");
  return wp <= tp && ws <= ts ? 0 : 1;
}

// The integer network: GPU (BankQ) vs CPU NetQ, which must agree bit for bit.
template <class R> static int runq(bool self, int d, int nets, int N, Api api, int units, bool v43) {
  if (d < 1 || d > 128 || nets < 1 || N < 1 || !valid_units(units)) { std::fprintf(stderr, "need 1 <= d <= 128, nets >= 1, T >= 1, units 8, 12 or 16\n"); return 1; }
  gpu::BankQ<R> bank; std::string why;
  if (!bank.open(api, 0, &why, options())) { std::printf("no GPU: %s\n", why.c_str()); return 77; }
  Rng rng(12345);
  std::vector<R> X((size_t)N * d), Y((size_t)nets * N), P;
  for (int t = 0; t < N; t++) {
    for (int i = 0; i < d; i++) X[(size_t)t * d + i] = (R)(2 * rng.uni() - 1);
    const R* x = &X[(size_t)t * d];
    R y = std::sin(3 * x[0]) * std::cos(2 * x[1 % d]) + (R)0.5 * x[2 % d] * x[3 % d];
    for (int n = 0; n < nets; n++) Y[(size_t)n * N + t] = n % 3 == 0 ? y : (n % 3 == 1 ? -y : (R)0.5 * y);
  }
  const double g0 = d > 16 ? 0.02 : 0.2, g1 = d > 16 ? 0.02 : 0.05;
  Params prm = v43 ? Params::v43(g0, g1) : Params::v5(g0, g1); prm.units = units;
  std::vector<NetQ<R>> nn(nets);
  for (int n = 0; n < nets; n++) nn[n].init(d, 7 + n, prm);
  if (!bank.upload(nn, &why)) { std::printf("upload failed: %s\n", why.c_str()); return 1; }
  std::printf("%s, image %s, integer net (int%d), %s, %d units, scratch %zu bytes per network in global memory\n", bank.device_name().c_str(),
              bank.image().c_str(), prm.wbits, v43 ? "4.3 structure" : "v5", units, bank.scratch_bytes());
  auto t0 = std::chrono::steady_clock::now();
  if (!bank.run(X, Y, P, N, &why)) { std::printf("run failed: %s\n", why.c_str()); return 1; }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("GPU: %d networks x %d samples in %.2f ms = %.2f us per sample (all networks in parallel; includes copies)\n", nets, N, ms, 1000.0 * ms / N);
  if (!self) return 0;
  std::vector<NetQ<R>> got;
  if (!bank.download(got, &why)) { std::printf("download failed: %s\n", why.c_str()); return 1; }
  long bp = 0, bs = 0;
  for (int n = 0; n < nets; n++) {
    for (int t = 0; t < N; t++) bp += nn[n].step(&X[(size_t)t * d], Y[(size_t)n * N + t]) != P[(size_t)n * N + t];
    bs += got[n].t != nn[n].t || got[n].pruned() != nn[n].pruned();
    for (int l = 0; l < 3; l++) {
      const auto &A = got[n].L[l], &B = nn[n].L[l];
      for (int k = 0; k < units; k++) {
        for (int i = 0; i < A.D; i++) { bs += A.gate_weight(k, i) != B.gate_weight(k, i); for (int b = 0; b < 4; b++) bs += A.hyperplane(k, b, i) != B.hyperplane(k, b, i); }
        for (int j = 0; j < A.N1; j++) {
          bs += A.active(k, j) != B.active(k, j) || A.activity(k, j) != B.activity(k, j);
          for (int c = 0; c < 16; c++) bs += A.rule(k, c, j) != B.rule(k, c, j);
        }
      }
    }
  }
  std::printf("selftest %s integer d=%d nets=%d T=%d units=%d %s: predictions differing %ld, state integers differing %ld (%s inputs)\n", gpu::api_name(api),
              d, nets, N, units, v43 ? "4.3 structure" : "v5", bp, bs, sizeof(R) == 8 ? "float64" : "float32");
  return bp == 0 && bs == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "info";
  if (mode == "info") return info();
  if ((mode != "selftest" && mode != "bench") || argc < 5) {
    std::fprintf(stderr, "usage: hgln_gpu info | selftest|bench d nets T [cuda|hip] [f32] [v42] [units=8|12|16] [fanout] [v5] [int [v43]]\n");
    return 1;
  }
  const int d = std::atoi(argv[2]), nets = std::atoi(argv[3]), N = std::atoi(argv[4]);
  Api api = Api::cuda; bool f32 = false, v42 = false, fanout = false, v5 = false, iq = false, v43 = false; int units = 16; bool units_given = false;
  for (int i = 5; i < argc; i++) {
    if (!std::strcmp(argv[i], "hip")) api = Api::hip;
    else if (!std::strcmp(argv[i], "cuda")) api = Api::cuda;
    else if (!std::strcmp(argv[i], "f32")) f32 = true;
    else if (!std::strcmp(argv[i], "v42")) v42 = true;
    else if (!std::strcmp(argv[i], "fanout")) fanout = true;
    else if (!std::strcmp(argv[i], "v5")) v5 = true;
    else if (!std::strcmp(argv[i], "int")) iq = true;
    else if (!std::strcmp(argv[i], "v43")) v43 = true;
    else if (!std::strncmp(argv[i], "units=", 6)) { units = std::atoi(argv[i] + 6); units_given = true; }
    else { std::fprintf(stderr, "unknown option '%s'\n", argv[i]); return 1; }
  }
  const bool self = mode == "selftest";
  if (iq) {
    if (!units_given) units = v43 ? 16 : 8;
    if (f32) return runq<float>(self, d, nets, N, api, units, v43);
    int r = runq<double>(self, d, nets, N, api, units, v43);
    if (self && r == 0) r = runq<float>(self, d, nets, N, api, units, v43);
    return r;
  }
  if (f32) return run<float>(self, d, nets, N, api, v42, units, fanout, v5);
  int r = run<double>(self, d, nets, N, api, v42, units, fanout, v5);
  if (self && r == 0) r = run<float>(self, d, nets, N, api, v42, units, fanout, v5);   // selftest checks both real types
  return r;
}
