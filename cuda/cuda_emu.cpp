// SPDX-License-Identifier: AGPL-3.0-or-later
// cuda_emu.cpp - runs the CUDA kernel source on the CPU (each block = a set of std::threads with a barrier) and
// compares it with the reference C++ network. Tests the kernel's arithmetic, indexing, phase ordering and (to the
// extent a run can show) its synchronisation. It does NOT test GPU behaviour. Copyright (C) Odin Loch. GNU AGPL v3+.
//   cuda_emu d nets T threads [shared] [competence] [far] [units] [fanout]
//     shared = 1: per-block scratch in emulated shared memory, pre-filled with NaN (catches any read of scratch before
//                 it is written); 0 (default): global
//     competence = 1: version 4.2 competence-weighted mixture (Params::v42); 0 (default): version 4.3 plain average
//     far = 1: start layer 1's hyperplanes at |H|^2 just under 1e6 and layer 2's just over 1e-6, so the conditional
//              layer renormalisation (norm out of range) runs; T > 4096 also crosses the periodic renormalisation
//     units = Params::units (8, 12 or 16; default 16); fanout = 1: Params::fanout (layer 3 reads layer 1)
//     v5 = version 5.0 options: 1 the v5 defaults (latch 0.1, latch credit 0.5, gate_err 2, glob 0.95, top-6 corners); 2 latch 0.1
//          without credit, gate_err 1; 3 gate_err 2 without the latch; 4 = 1 with all 16 corners; 5 top-8 corners only; 0 (default) none (4.3)
// float64 and float32 must match the C++ library exactly (0 difference), predictions and the full state including |H|^2.
#define HGLN_EMULATE
#include <barrier>
#include <thread>
#include <memory>
#include <cstdio>
#include <cstdlib>
static thread_local int hg_tid_v = 0, hg_nt_v = 1, hg_bid_v = 0;
static thread_local std::barrier<>* hg_bar = nullptr;
static thread_local double* hg_smem_v = nullptr;
#define HG_KERNEL static
#define HG_DEV static inline
#define HG_HD
#define HG_TID (hg_tid_v)
#define HG_NT (hg_nt_v)
#define HG_BID (hg_bid_v)
#define HG_SYNC() hg_bar->arrive_and_wait()
#define HG_SMEM_DECL
#define HG_SMEM (reinterpret_cast<hreal*>(hg_smem_v))
#include "hgln_host.h"
using hgln::Net;
int main(int argc, char** argv) {
  int d = argc > 1 ? atoi(argv[1]) : 8, nets = argc > 2 ? atoi(argv[2]) : 3, T = argc > 3 ? atoi(argv[3]) : 1500, NT = argc > 4 ? atoi(argv[4]) : 8, SH = argc > 5 ? atoi(argv[5]) : 0,
      CM = argc > 6 ? atoi(argv[6]) : 0, FAR = argc > 7 ? atoi(argv[7]) : 0, UN = argc > 8 ? atoi(argv[8]) : 16, FO = argc > 9 ? atoi(argv[9]) : 0,
      V5 = argc > 10 ? atoi(argv[10]) : 0;
  if (!hg_valid_units(UN) || d < 1 || d > HG_MAXN) { fprintf(stderr, "cuda_emu: need units 8, 12 or 16 and 1 <= d <= %d\n", HG_MAXN); return 2; }
  HgDims g = hg_dims(d, UN); hgln::Rng rng(12345);
  std::vector<hreal> X((size_t)T * d), Y((size_t)nets * T);
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < d; i++) X[(size_t)t * d + i] = 2 * rng.uni() - 1;
    hreal y = std::sin(3 * X[(size_t)t * d]) * std::cos(2 * X[(size_t)t * d + 1]) + 0.5 * X[(size_t)t * d + 2 % d] * X[(size_t)t * d + 3 % d];
    for (int n = 0; n < nets; n++) Y[(size_t)n * T + t] = n % 3 == 0 ? y : (n % 3 == 1 ? -y : 0.5 * y);   // different targets per network
  }
  hgln::Params prm = hgln::Params::v43(d > 16 ? 0.02 : 0.2, d > 16 ? 0.02 : 0.05); prm.competence = CM != 0; prm.units = UN; prm.fanout = FO != 0;
  if (V5 == 1) { prm.latch = 0.1; prm.latch_credit = 0.5; prm.gate_err = 2; prm.glob = 0.95; prm.corners = 6; }
  else if (V5 == 2) { prm.latch = 0.1; prm.gate_err = 1; }
  else if (V5 == 3) prm.gate_err = 2;
  else if (V5 == 4) { prm.latch = 0.1; prm.latch_credit = 0.5; prm.gate_err = 2; prm.glob = 0.95; }   // v5 without top-K
  else if (V5 == 5) prm.corners = 8;   // 4.3 with top-8 corners
  std::vector<Net<hreal>> init(nets), ref(nets);
  for (int n = 0; n < nets; n++) {
    init[n].init(d, 7 + n, prm);
    if (FAR) for (int l = 0; l < 2; l++) {                   // same directions, norms near the edges of (1e-6, 1e6)
      std::vector<hreal> Hr((size_t)4 * UN * g.D), q(4 * UN);
      init[n].L[l].get_hyperplanes_raw(Hr.data(), q.data());
      for (auto& v : Hr) v *= l == 0 ? (hreal)999.9999 : (hreal)1.000001e-3;
      init[n].L[l].set_hyperplanes(Hr.data());
    }
    ref[n] = init[n];
  }
  HgHostState st; hg_pack(g, init, st, !SH);
  std::vector<hreal> P((size_t)nets * T);
  for (int b = 0; b < nets; b++) {                          // blocks run one after another; threads inside a block run concurrently
    std::barrier<> bar(NT); std::vector<std::thread> th;
    std::vector<double> smem(SH ? (g.scs * sizeof(hreal) + sizeof(double) - 1) / sizeof(double) : 0);
    for (size_t i = 0; i < smem.size(); i++) smem[i] = std::nan("");   // NaN bytes for float too: any early read shows
    for (int k = 0; k < NT; k++) th.emplace_back([&, k] {
      hg_tid_v = k; hg_nt_v = NT; hg_bid_v = b; hg_bar = &bar; hg_smem_v = smem.data();
      hgln_run(g, st.H.data(), st.Q.data(), st.p.data(), st.W.data(), st.M.data(), st.A.data(), st.m.data(), SH ? nullptr : st.scratch.data(),
               X.data(), Y.data(), P.data(), st.t.data(), T, (hreal)prm.glr0, (hreal)prm.glr1, (hreal)prm.tau, (hreal)prm.glob, (hreal)prm.mu, prm.warm,
               st.competence, st.units, st.fanout, st.latch, st.credit, st.gate_err, st.corners); });
    for (auto& t : th) t.join();
  }
  double worstP = 0, worstS = 0;
  for (int n = 0; n < nets; n++) {
    for (int t = 0; t < T; t++) { hreal pr = ref[n].predict(&X[(size_t)t * d]); ref[n].learn(Y[(size_t)n * T + t]); worstP = std::max(worstP, std::fabs((double)pr - (double)P[(size_t)n * T + t])); }
    worstS = std::max(worstS, hg_state_diff(g, st, n, ref[n]));
  }
  // with far = 1 and T < 4096, layers 1-2 can only be back to |H|^2 in (0.01, 100) if the conditional renormalisation ran
  int renormed = 0;
  if (FAR) for (int n = 0; n < nets; n++) for (int l = 0; l < 2; l++) {
    bool unit = true; for (int k = 0; k < UN; k++) for (int b = 0; b < 4; b++) { const double q = (double)ref[n].L[l].hyperplane_sqnorm(k, b); unit &= q > 0.01 && q < 100; }
    renormed += unit;
  }
  char farmsg[64] = "";
  if (FAR) snprintf(farmsg, sizeof farmsg, ", far: %d/%d layers renormalised", renormed, 2 * nets);
  printf("cuda_emu d=%d nets=%d T=%d threads=%d real=%s scratch=%s (%ld bytes/block) competence=%d units=%d %s%s v5=%d: max |prediction diff| = %.3g, max |state diff| = %.3g\n",
         d, nets, T, NT, sizeof(hreal) == 4 ? "float32" : "float64", SH ? "shared" : "global", (long)(g.scs * sizeof(hreal)), CM, UN, FO ? "fanout" : "chain", farmsg, V5, worstP, worstS);
  return worstP == 0 && worstS == 0 ? 0 : 1;
}
