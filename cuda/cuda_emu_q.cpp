// SPDX-License-Identifier: AGPL-3.0-or-later
// cuda_emu_q.cpp - runs the integer GPU kernel (cuda/hgln_q_kernel.h) on the CPU (each block = a set of std::threads
// with a barrier) and compares it with the C++ integer network hgln::NetQ: predictions and the whole state must be
// bit-identical. It does NOT test GPU behaviour. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
//   cuda_emu_q d nets T threads [units] [bits] [mode]
//     units: 8, 12 or 16 (default 8); bits: wbits = abits, 12..16 (default 16)
//     mode 0: the v5 defaults (fan-out, latch + credit, global-error gates, top-6 corners); 1: the 4.3 structure (chain,
//          plain gates, mixed gate error, all 16 corners, 14-bit activations); 2: chain latch, credit 1, top-1;
//          3: chain, plain gates, own-error gates, top-8
#define HGLN_EMULATE
#include <barrier>
#include <thread>
#include <cstdio>
#include <cstdlib>
static thread_local int hg_tid_v = 0, hg_nt_v = 1, hg_bid_v = 0;
static thread_local std::barrier<>* hg_bar = nullptr;
#define HG_KERNEL static
#define HG_DEV static inline
#define HG_HD
#define HG_TID (hg_tid_v)
#define HG_NT (hg_nt_v)
#define HG_BID (hg_bid_v)
#define HG_SYNC() hg_bar->arrive_and_wait()
#define HG_SMEM_DECL
#define HG_SMEM nullptr
#include "hgln_kernel.h"
#include "hgln_q_kernel.h"
#include "hgln_q_host.h"
using hgln::NetQ; using hgln::Params;
[[maybe_unused]] static void (*const hg_float_kernel)(HgDims, hreal*, hreal*, hreal*, hreal*, hreal*, hreal*, hreal*, hreal*, const hreal*,
    const hreal*, hreal*, long*, int, hreal, hreal, hreal, hreal, hreal, int, int, int, int, hreal, hreal, int, int) = hgln_run;   // included, unused here
int main(int argc, char** argv) {
  const int d = argc > 1 ? atoi(argv[1]) : 8, nets = argc > 2 ? atoi(argv[2]) : 2, T = argc > 3 ? atoi(argv[3]) : 1500, NT = argc > 4 ? atoi(argv[4]) : 8,
            UN = argc > 5 ? atoi(argv[5]) : 8, BITS = argc > 6 ? atoi(argv[6]) : 16, MODE = argc > 7 ? atoi(argv[7]) : 0;
  if (!hg_valid_units(UN) || d < 1 || d > HG_MAXN) { fprintf(stderr, "cuda_emu_q: need units 8, 12 or 16 and 1 <= d <= %d\n", HG_MAXN); return 2; }
  const double g0 = d > 16 ? 0.02 : 0.2, g1 = d > 16 ? 0.02 : 0.05;
  Params P = Params::v5(g0, g1);
  if (MODE == 1) { P = Params::v43(g0, g1); P.abits = 14; }
  else if (MODE == 2) { P.fanout = false; P.latch_credit = 1.0; P.corners = 1; }
  else if (MODE == 3) { P = Params::v43(g0, g1); P.corners = 8; P.gate_err = 1; }
  P.units = UN; P.wbits = BITS; if (MODE != 1) P.abits = BITS;
  const HgQDims g = hq_dims(d, UN); hgln::Rng rng(12345);
  std::vector<double> X((size_t)T * d), Y((size_t)nets * T);
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < d; i++) X[(size_t)t * d + i] = 2 * rng.uni() - 1;
    const double y = std::sin(3 * X[(size_t)t * d]) * std::cos(2 * X[(size_t)t * d + 1 % d]) + 0.5 * X[(size_t)t * d + 2 % d] * X[(size_t)t * d + 3 % d];
    for (int n = 0; n < nets; n++) Y[(size_t)n * T + t] = n % 3 == 0 ? y : (n % 3 == 1 ? -y : 0.5 * y);
  }
  std::vector<NetQ<double>> ref(nets);
  for (int n = 0; n < nets; n++) ref[n].init(d, 7 + n, P);
  HqHostState st; hq_pack(g, ref, st);
  const HgQCfg C = hq_cfg(ref[0]);
  const std::vector<short> lut = hq_sig_table(); const std::vector<long long> ex = hq_exp_table();
  std::vector<double> Pd((size_t)nets * T);
  for (int b = 0; b < nets; b++) {                          // blocks one after another; threads of a block concurrently
    std::barrier<> bar(NT); std::vector<std::thread> th;
    for (int k = 0; k < NT; k++) th.emplace_back([&, k] {
      hg_tid_v = k; hg_nt_v = NT; hg_bid_v = b; hg_bar = &bar;
      hglnq_run(g, C, st.H.data(), st.p.data(), st.W.data(), st.M.data(), st.A.data(), st.scratch.data(), lut.data(), ex.data(),
                X.data(), Y.data(), Pd.data(), st.t.data(), T); });
    for (auto& t : th) t.join();
  }
  long badP = 0, badS = 0;
  for (int n = 0; n < nets; n++) {
    for (int t = 0; t < T; t++) { const double pr = ref[n].predict(&X[(size_t)t * d]); ref[n].learn(Y[(size_t)n * T + t]); badP += pr != Pd[(size_t)n * T + t]; }
    const long b = hq_state_diff(g, st, n, ref[n]); badS += b < 0 ? 1 << 30 : b;
  }
  // round trip: the device layout back into a fresh NetQ continues exactly like the CPU network
  long badR = 0;
  { NetQ<double> back; back.init(d, 99, P); hq_unpack(g, st, 0, back); NetQ<double> cpu = ref[0];
    for (int t = 0; t < 300; t++) { std::vector<double> x(d); for (auto& v : x) v = 2 * rng.uni() - 1;
      badR += back.predict(x.data()) != cpu.predict(x.data()); back.learn(std::cos(2 * x[0])); cpu.learn(std::cos(2 * x[0])); } }
  printf("cuda_emu_q d=%d nets=%d T=%d threads=%d units=%d bits=%d/%d corners=%d %s%s%s gate_err=%d: predictions differing %ld, state integers differing %ld, after round trip %ld\n",
         d, nets, T, NT, UN, P.wbits, P.abits, P.corners, P.fanout ? "fanout" : "chain", P.latch > 0 ? " latch" : "", P.latch_credit > 0 && P.latch > 0 ? "+credit" : "",
         P.gate_err, badP, badS, badR);
  return badP == 0 && badS == 0 && badR == 0 ? 0 : 1;
}
