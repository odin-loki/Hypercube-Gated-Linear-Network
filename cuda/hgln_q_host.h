// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_q_host.h - host helpers for the integer kernel (cuda/hgln_q_kernel.h): its constants from a NetQ, and the
// network state copied into / out of the kernel's dense layout (cuda/hgln_q_dims.h) through the LayerQ accessors and
// NetQ::import_dense. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#pragma once
#include "hgln_q_dims.h"
#include "hgln_q.hpp"
#include <cstdint>
#include <vector>

template <class R> inline HgQCfg hq_cfg(const hgln::NetQ<R>& n) {
  const hgln::qspec::Fmt& F = n.C.F; HgQCfg c;
  c.AB = F.AB; c.WB = F.WB; c.FA = F.FA; c.FF = F.FF; c.FH = F.FH; c.FP = F.FP; c.MA = F.MA; c.MW = F.MW; c.ONE = F.ONE;
  c.K = n.C.K; c.latch = n.C.latch ? 1 : 0; c.credit = n.C.credit ? 1 : 0; c.gate_err = n.P.gate_err; c.fanout = n.P.fanout ? 1 : 0;
  c.warm = n.P.warm; c.units = n.units(); c.th14 = n.C.th14; c.cscale = n.C.cscale;
  c.G = hgln::qspec::round_param(n.P.glob, 16); c.Mq = hgln::qspec::round_param(n.P.mu, 16);
  c.g0 = hgln::qspec::round_param(n.P.glr0, 24); c.g1 = hgln::qspec::round_param(n.P.glr1, 24); c.tau = n.P.tau;
  return c;
}
inline std::vector<short> hq_sig_table() { return std::vector<short>(hgln::qspec::kSigLut, hgln::qspec::kSigLut + 1025); }
inline std::vector<long long> hq_exp_table() { return std::vector<long long>(hgln::qspec::kExpNeg, hgln::qspec::kExpNeg + 22); }

struct HqHostState { std::vector<int32_t> H, p, W, M, A; std::vector<long long> scratch; std::vector<long> t; };
// Pack the networks (same d and Params) into the dense layout; with_scratch: also the global scratch.
template <class R> inline void hq_pack(const HgQDims& g, const std::vector<hgln::NetQ<R>>& nets, HqHostState& s, bool with_scratch = true) {
  const size_t N = nets.size(); const int U = g.U, D = g.D;
  s.H.assign(N * g.Hs, 0); s.p.assign(N * g.ps, 0); s.W.assign(N * g.Ws, 0); s.M.assign(N * g.Ms, 0); s.A.assign(N * g.Ms, 0);
  s.scratch.assign(with_scratch ? N * (size_t)g.scs : 0, 0); s.t.assign(N, 0);
  for (size_t n = 0; n < N; n++) {
    s.t[n] = (long)nets[n].t;
    for (int l = 0; l < 3; l++) {
      const auto& L = nets[n].L[l]; const int N1 = g.N1[l];
      for (int k = 0; k < U; k++) {
        for (int i = 0; i < D; i++) {
          s.p[n * g.ps + ((size_t)l * U + k) * D + i] = L.gate_weight(k, i);
          for (int b = 0; b < 4; b++) s.H[n * g.Hs + (((size_t)l * U + k) * 4 + b) * D + i] = L.hyperplane(k, b, i);
        }
        for (int j = 0; j < N1; j++) {
          s.M[n * g.Ms + g.moff[l] + (size_t)k * N1 + j] = L.active(k, j) ? 1 : 0;
          s.A[n * g.Ms + g.moff[l] + (size_t)k * N1 + j] = L.activity(k, j);
          for (int c = 0; c < 16; c++) s.W[n * g.Ws + g.woff[l] + ((size_t)k * 16 + c) * N1 + j] = L.rule(k, c, j);
        }
      }
    }
  }
}
// Copy network n of the dense state into a NetQ initialised with the same d and Params (sample count included).
template <class R> inline void hq_unpack(const HgQDims& g, const HqHostState& s, size_t n, hgln::NetQ<R>& net) {
  const bool pruned = s.t[n] >= hg_prune_time(net.P.warm);   // the kernel prunes once, at t == the prune time
  for (int l = 0; l < 3; l++)
    net.import_dense(l, &s.H[n * g.Hs + (size_t)l * g.U * 4 * g.D], &s.p[n * g.ps + (size_t)l * g.U * g.D], &s.W[n * g.Ws + g.woff[l]],
                     &s.M[n * g.Ms + g.moff[l]], &s.A[n * g.Ms + g.moff[l]], pruned);
  net.t = s.t[n]; net.pruned_ = pruned;
}
// Number of stored integers of network n that differ from a CPU NetQ (0 = bit-identical): H, p, M, the active W and,
// before the prune, the activity trace; and the sample count.
template <class R> inline long hq_state_diff(const HgQDims& g, const HqHostState& s, size_t n, const hgln::NetQ<R>& net) {
  if (net.units() != g.U || net.d != g.d) return -1;
  long bad = s.t[n] != (long)net.t;
  const int U = g.U, D = g.D;
  for (int l = 0; l < 3; l++) {
    const auto& L = net.L[l]; const int N1 = g.N1[l];
    for (int k = 0; k < U; k++) {
      for (int i = 0; i < D; i++) {
        bad += s.p[n * g.ps + ((size_t)l * U + k) * D + i] != L.gate_weight(k, i);
        for (int b = 0; b < 4; b++) bad += s.H[n * g.Hs + (((size_t)l * U + k) * 4 + b) * D + i] != L.hyperplane(k, b, i);
      }
      for (int j = 0; j < N1; j++) {
        const bool act = L.active(k, j);
        bad += (s.M[n * g.Ms + g.moff[l] + (size_t)k * N1 + j] != 0) != act;
        if (!L.pruned) bad += s.A[n * g.Ms + g.moff[l] + (size_t)k * N1 + j] != L.activity(k, j);
        if (act) for (int c = 0; c < 16; c++) bad += s.W[n * g.Ws + g.woff[l] + ((size_t)k * 16 + c) * N1 + j] != L.rule(k, c, j);
      }
    }
  }
  return bad;
}
