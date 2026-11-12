// SPDX-License-Identifier: AGPL-3.0-or-later
// test_gpu.cpp - hgln::gpu::Bank end to end: Net -> Bank -> FORGE gpu::cuda_hip runtime -> driver -> hgln_run ->
// results, compared with the CPU library sample by sample. Under ctest the driver is HGLN's mock (tests/mock_hgln.c),
// which runs the real kernel source on the CPU. Float64 and float32 must agree EXACTLY (0 difference): the kernel
// does the library's IEEE operations in the same order, with the same exp (hgln_exp) and gate (hgln_gate). On a
// machine with a real GPU and no mock on the library path the same program tests the hardware, where the same holds
// (checked on an RTX 3090).
//   hgln_test_gpu cuda|hip       all checks on that API (d = 8, 64, 100; float64 and float32; shared and global
//                                scratch; both mixtures: 4.3 plain average and 4.2 competence; several calls with
//                                state resident; download, re-upload, continue; across the prune, the conditional
//                                renormalisation of an out-of-range layer and the renormalisation at t = 4096)
//   hgln_test_gpu cuda|hip units=N [fanout] [v5=N]   a shorter set of the same checks for Params::units = N (8, 12
//                                or 16) and/or Params::fanout (layer 3 reads layer 1) and/or the version 5.0 options
//                                v5=N (1 the v5 options incl. top-6 corners, 2 latch + own-error gates, 3 gate_err 2
//                                only, 4 the v5 options without top-K, 5 top-8 corners only; see set_v5 below):
//                                d = 8, 64, 100; both real types, both scratch placements, both mixtures; across the
//                                prune, the out-of-range renormalisation and the renormalisation at t = 4096
//                                (the integer network has its own mode, `int`, listed next)
//   hgln_test_gpu cuda|hip int   the integer network (NetQ) through BankQ: bit-identical predictions and state on any
//                                driver (the kernel is integer arithmetic): v5 and the 4.3 structure, 8/12/16 units,
//                                d = 8, 64, 100, both real types, several calls, download / re-upload, across the prune
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
  if (a.t != b.t || a.d != b.d || a.units() != b.units() || a.P.fanout != b.P.fanout) return INFINITY;
  for (int l = 0; l < 3; l++) {
    const auto &A = a.L[l], &B = b.L[l];
    if (A.pruned != B.pruned || A.has_competence != B.has_competence) return INFINITY;
    for (int k = 0; k < A.U; k++) {
      for (int q = 0; q < 4; q++) {                       // the exact state: raw rows and squared norms
        upd(A.hyperplane_sqnorm(k, q), B.hyperplane_sqnorm(k, q));
        for (int i = 0; i < A.D; i++) { upd(A.hyperplane_raw(k, q, i), B.hyperplane_raw(k, q, i)); upd(A.hyperplane(k, q, i), B.hyperplane(k, q, i)); }
      }
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

// competence: Params::competence (4.2 mixture). far: start layers 1 and 2 with |H|^2 just under 1e6 / just over 1e-6
// (same directions), so the update soon takes them out of range and the layer renormalisation runs.
// units, fanout: Params::units and Params::fanout. g_v5 (option v5=N): the version 5.0 options, as cuda/cuda_emu.cpp:
// 1 the v5 options (latch 0.1 + latch credit 0.5 + gate_err 2 + glob 0.95 + top-6 corners); 2 latch 0.1 + gate_err 1;
// 3 gate_err 2 only; 4 = 1 with all 16 corners; 5 top-8 corners only.
static int g_v5 = 0;
static void set_v5(Params& p, int v5) {
  if (v5 == 1 || v5 == 4) { p.latch = 0.1; p.latch_credit = 0.5; p.gate_err = 2; p.glob = 0.95; p.corners = v5 == 1 ? 6 : 16; }
  else if (v5 == 2) { p.latch = 0.1; p.gate_err = 1; }
  else if (v5 == 3) p.gate_err = 2;
  else if (v5 == 5) p.corners = 8;
}
template <class T> static void one(Api api, int d, int nets, int N, gpu::Scratch scratch, int threads, const std::vector<int>& chunks,
                                   bool competence = false, bool far = false, int units = 16, bool fanout = false) {
  const bool f64 = sizeof(T) == 8;
  const double tolP = 0, tolS = 0;   // the kernel and the library do the same IEEE operations (hgln_gate, hgln_exp): exact
  Params prm = Params::v43(d > 16 ? 0.02 : 0.2, d > 16 ? 0.02 : 0.05); prm.competence = competence; prm.units = units; prm.fanout = fanout;
  set_v5(prm, g_v5);
  std::vector<Net<T>> init(nets);
  for (int n = 0; n < nets; n++) {
    init[n].init(d, 7 + n, prm);
    if (far) for (int l = 0; l < 2; l++) {
      std::vector<T> H((size_t)4 * units * init[n].L[l].D), q(4 * units);
      init[n].L[l].get_hyperplanes_raw(H.data(), q.data());
      for (auto& v : H) v *= l == 0 ? (T)999.9999 : (T)1.000001e-3;
      init[n].L[l].set_hyperplanes(H.data());
    }
  }
  std::vector<Net<T>> ref = init;
  Stream<T> st(d, nets, N, 12345 + d);

  gpu::Bank<T> bank; std::string why;
  gpu::Options opt; opt.threads = threads; opt.scratch = scratch;
  if (const char* e = std::getenv("HGLN_GPU_KERNEL")) opt.kernel = !std::strcmp(e, "block") ? gpu::Kernel::block : !std::strcmp(e, "warp") ? gpu::Kernel::warp : gpu::Kernel::automatic;
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
      const unsigned W = (unsigned)bank.nets_per_block(), eg = bank.warp_kernel() ? ((unsigned)nets + W - 1) / W : (unsigned)nets,
                     eb = bank.warp_kernel() ? 32 * W : (unsigned)threads;
      CHECK(g == eg && b == eb && real == sizeof(T), "launch grid %u block %u real %u (%s kernel)", g, b, real, bank.warp_kernel() ? "warp" : "block");
      CHECK(sm == (bank.shared_scratch() ? (bank.warp_kernel() ? W : 1) * bank.scratch_bytes() : 0), "launch smem %u", sm);
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
  if (far && N < 4096)   // the start norms only come back to (0.01, 100) through the out-of-range renormalisation
    for (int n = 0; n < nets; n++) for (int l = 0; l < 2; l++) for (int k = 0; k < units; k++) for (int q = 0; q < 4; q++) {
      const double v = (double)ref[n].L[l].hyperplane_sqnorm(k, q);
      CHECK(v > 0.01 && v < 100, "far: layer %d was not renormalised (|H|^2 = %g)", l, v);
    }
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
  std::printf("%s %s d=%d nets=%d T=%d calls=%d threads=%d scratch=%s(%zu B) kernel=%s/%d image=%s competence=%d units=%d %s%s v5=%d: max |P diff| = %.3g, state diff mid = %.3g, end = %.3g, after re-upload = %.3g\n",
              gpu::api_name(api), f64 ? "float64" : "float32", d, nets, N, calls, threads, bank.shared_scratch() ? "shared" : "global",
              bank.scratch_bytes(), bank.warp_kernel() ? "warp" : "block", bank.nets_per_block(), bank.image().c_str(), competence ? 1 : 0, units, fanout ? "fanout" : "chain", far ? " far" : "", g_v5, wP, wMid, wS, wR);
  CHECK(wP <= tolP && wMid <= tolS && wS <= tolS && wR <= std::max(tolP, tolS), "differences above tolerance (P %.3g, state %.3g)", tolP, tolS);
  std::fflush(stdout);
}

// ---- the integer network through BankQ: everything must be bit-identical
template <class R> static long qstate_diff(const NetQ<R>& a, const NetQ<R>& b) {
  if (a.t != b.t || a.d != b.d || a.units() != b.units() || a.pruned() != b.pruned()) return 1L << 40;
  long bad = 0;
  for (int l = 0; l < 3; l++) {
    const auto &A = a.L[l], &B = b.L[l];
    for (int k = 0; k < A.U; k++) {
      for (int i = 0; i < A.D; i++) { bad += A.gate_weight(k, i) != B.gate_weight(k, i); for (int q = 0; q < 4; q++) bad += A.hyperplane(k, q, i) != B.hyperplane(k, q, i); }
      for (int j = 0; j < A.N1; j++) { bad += A.active(k, j) != B.active(k, j) || A.activity(k, j) != B.activity(k, j); for (int c = 0; c < 16; c++) bad += A.rule(k, c, j) != B.rule(k, c, j); }
    }
  }
  return bad;
}
template <class R> static void oneq(Api api, int d, int nets, int N, int threads, const std::vector<int>& chunks, int units, bool v43) {
  const double g0 = d > 16 ? 0.02 : 0.2, g1 = d > 16 ? 0.02 : 0.05;
  Params prm = v43 ? Params::v43(g0, g1) : Params::v5(g0, g1); prm.units = units;
  std::vector<NetQ<R>> init(nets); for (int n = 0; n < nets; n++) init[n].init(d, 7 + n, prm);
  std::vector<NetQ<R>> ref = init; Stream<R> st(d, nets, N, 4321 + d);
  gpu::BankQ<R> bank; std::string why; gpu::Options opt; opt.threads = threads;
  if (!bank.open(api, 0, &why, opt)) { CHECK(false, "BankQ open: %s", why.c_str()); return; }
  if (!bank.upload(init, &why)) { CHECK(false, "BankQ upload: %s", why.c_str()); return; }
  long bp = 0, bmid = 0; int s = 0, calls = 0; std::vector<R> x, y, P;
  for (size_t c = 0; c < chunks.size() && s < N; c++) {
    const int len = std::min(chunks[c], N - s); st.chunk(s, len, x, y);
    if (!bank.run(x, y, P, len, &why)) { CHECK(false, "BankQ run: %s", why.c_str()); return; }
    calls++;
    for (int i = 0; i < len; i++) for (int n = 0; n < nets; n++) bp += ref[n].step(&x[(size_t)i * d], y[(size_t)n * len + i]) != P[(size_t)n * len + i];
    s += len;
    if (c == 1) { std::vector<NetQ<R>> mid; if (!bank.download(mid, &why)) { CHECK(false, "download: %s", why.c_str()); return; }
      for (int n = 0; n < nets; n++) bmid += qstate_diff(mid[n], ref[n]); }
  }
  std::vector<NetQ<R>> got; if (!bank.download(got, &why)) { CHECK(false, "download: %s", why.c_str()); return; }
  long bend = 0; for (int n = 0; n < nets; n++) bend += qstate_diff(got[n], ref[n]);
  long bre = 0;
  {   // re-upload the downloaded networks and continue on both sides
    const int extra = 64; Stream<R> st2(d, nets, extra, 77 + d);
    if (!bank.upload(got, &why)) { CHECK(false, "re-upload: %s", why.c_str()); return; }
    st2.chunk(0, extra, x, y);
    if (!bank.run(x, y, P, extra, &why)) { CHECK(false, "run after re-upload: %s", why.c_str()); return; }
    for (int i = 0; i < extra; i++) for (int n = 0; n < nets; n++) bre += ref[n].step(&x[(size_t)i * d], y[(size_t)n * extra + i]) != P[(size_t)n * extra + i];
    std::vector<NetQ<R>> fin; if (!bank.download(fin, &why)) { CHECK(false, "download: %s", why.c_str()); return; }
    for (int n = 0; n < nets; n++) bre += qstate_diff(fin[n], ref[n]);
  }
  std::printf("%s integer (%s inputs) d=%d nets=%d T=%d calls=%d threads=%d image=%s units=%d %s: predictions differing %ld, state differing mid %ld, end %ld, after re-upload %ld\n",
              gpu::api_name(api), sizeof(R) == 8 ? "float64" : "float32", d, nets, N, calls, threads, bank.image().c_str(), units, v43 ? "4.3 structure" : "v5", bp, bmid, bend, bre);
  CHECK(bp == 0 && bmid == 0 && bend == 0 && bre == 0, "integer network: GPU and CPU must agree bit for bit");
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
  int units = 16; bool fanout = false, variant = false, integer = false;
  for (int i = 2; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "fanout") fanout = true;
    else if (a == "int") { integer = true; continue; }
    else if (a.rfind("units=", 0) == 0) units = std::atoi(a.c_str() + 6);
    else if (a.rfind("v5=", 0) == 0) g_v5 = std::atoi(a.c_str() + 3);
    else { std::printf("unknown option %s\n", a.c_str()); return 1; }
    variant = true;
  }
  if (integer) {   // the integer network through BankQ
    for (int u : {8, 12, 16}) {
      oneq<double>(api, 8, 2, 700, 32, chunks, u, false);
      oneq<float>(api, 64, 2, 600, 64, chunks, u, false);
    }
    oneq<double>(api, 100, 1, 560, 64, chunks, 8, false);
    oneq<double>(api, 8, 2, 700, 16, chunks, 16, true);                                  // the 4.3 structure
    oneq<float>(api, 8, 1, 4200, 32, {1, 200, 311, 3, 185, 3500}, 8, false);            // past t = 4096
    {   // networks with different Params cannot share a bank
      gpu::BankQ<double> b; std::string why; std::vector<NetQ<double>> mix(2); Params p1, p2; p2.corners = 8;
      mix[0].init(8, 1, p1); mix[1].init(8, 2, p2);
      CHECK(b.open(api, 0, &why) && !b.upload(mix, &why), "BankQ upload of networks with different Params must fail");
    }
    if (failures) { std::printf("%d FAILURES\n", failures); return 1; }
    std::printf("ALL GPU TESTS PASSED (%s, integer network)\n", gpu::api_name(api));
    return 0;
  }
  if (variant) {   // Params::units / Params::fanout / the v5 options: a shorter set of the same checks
    if (!valid_units(units)) { std::printf("units must be 8, 12 or 16\n"); return 1; }
    {   // networks with different units (or fanout, or v5 options) cannot share a bank
      gpu::Bank<double> b; std::string why;
      Params p1 = Params::v43(0.2, 0.05), p2 = p1; p1.units = units; p1.fanout = fanout; set_v5(p1, g_v5); p2 = p1; p2.units = units == 16 ? 8 : 16;
      std::vector<Net<double>> mix(2); mix[0].init(8, 1, p1); mix[1].init(8, 2, p2);
      CHECK(b.open(api, 0, &why) && !b.upload(mix, &why), "upload of networks with different units must fail");
      std::printf("mixed units: %s\n", why.c_str());
      p2 = p1; p2.fanout = !fanout; mix[1].init(8, 2, p2);
      CHECK(!b.upload(mix, &why), "upload of networks with different fanout must fail");
      if (g_v5) { p2 = p1; p2.latch = p1.latch > 0 ? 0 : 0.1; mix[1].init(8, 2, p2); CHECK(!b.upload(mix, &why), "upload of networks with different latch must fail"); }
    }
    one<double>(api, 8, 2, 700, S::shared, 32, chunks, false, false, units, fanout);
    one<double>(api, 64, 2, 600, S::global, 64, chunks, true, false, units, fanout);
    one<double>(api, 100, 1, 560, S::shared, 64, chunks, true, false, units, fanout);
    one<double>(api, 100, 1, 560, S::global, 64, chunks, false, false, units, fanout);
    one<float>(api, 8, 2, 700, S::global, 32, chunks, true, false, units, fanout);
    one<float>(api, 64, 1, 600, S::shared, 64, chunks, false, false, units, fanout);
    one<double>(api, 8, 2, 300, S::shared, 32, {1, 150, 149}, true, true, units, fanout);                     // out-of-range renormalisation
    one<double>(api, 8, 1, 4200, S::global, 32, {1, 200, 311, 3, 185, 3500}, false, false, units, fanout);    // past t = 4096
    if (failures) { std::printf("%d FAILURES\n", failures); return 1; }
    std::printf("ALL GPU TESTS PASSED (%s, units=%d %s v5=%d)\n", gpu::api_name(api), units, fanout ? "fanout" : "chain", g_v5);
    return 0;
  }
  for (bool cm : {false, true})
    for (S sc : {S::shared, S::global}) {
      one<double>(api, 8, 3, 700, sc, 32, chunks, cm);
      one<double>(api, 64, 2, 600, sc, 64, chunks, cm);
      one<double>(api, 100, 2, 560, sc, 64, chunks, cm);
      one<float>(api, 8, 3, 700, sc, 32, chunks, cm);
      one<float>(api, 64, 2, 600, sc, 64, chunks, cm);
      one<float>(api, 100, 2, 560, sc, 64, chunks, cm);
    }
  one<double>(api, 64, 2, 300, S::automatic, 256, {120, 180});   // the default block size, automatic placement
  one<float>(api, 8, 4, 300, S::automatic, 256, {300});
  for (bool cm : {false, true}) {
    one<double>(api, 8, 2, 300, S::shared, 32, {1, 150, 149}, cm, true);              // out-of-range layer renormalisation
    one<double>(api, 8, 1, 4200, S::shared, 32, {1, 200, 311, 3, 185, 3500}, cm);       // past the renormalisation at 4096
  }
  one<float>(api, 8, 1, 4200, S::global, 32, {600, 3600}, false);
  if (failures) { std::printf("%d FAILURES\n", failures); return 1; }
  std::printf("ALL GPU TESTS PASSED (%s)\n", gpu::api_name(api));
  return 0;
}
