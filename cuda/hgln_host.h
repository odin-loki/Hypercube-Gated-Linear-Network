// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_host.h - host helpers: copy a CPU hgln::Net (any training state) into / out of the flat kernel layout.
// The C++ library stores its arrays in a SIMD layout and drops pruned columns; these helpers convert through the
// Layer accessors, so the kernel keeps the simple dense layout. Copyright (C) Odin Loch. GNU AGPL v3.0 or later.
#pragma once
#include "hgln_kernel.h"
#include "hgln.hpp"
#include <vector>
struct HgHostState { std::vector<hreal> H, Q, p, W, M, A, m, scratch; std::vector<long> t; int competence = 0, units = 16, fanout = 0; };
// with_scratch = false: the kernel will use dynamic shared memory, so no global scratch is allocated.
// H is packed raw (unnormalised) with its squared norms Q, the exact state the library keeps, so the kernel continues
// bit for bit. s.competence, s.units and s.fanout are the nets' Params::competence, units and fanout (all nets must
// agree; they are the kernel's arguments). g must be hg_dims(d, units).
inline void hg_pack(const HgDims& g, const std::vector<hgln::Net<hreal>>& nets, HgHostState& s, bool with_scratch = true) {
  size_t N = nets.size();
  s.H.assign(N * g.Hs, 0); s.Q.assign(N * g.Qs, 0); s.p.assign(N * g.ps, 0); s.W.assign(N * g.Ws, 0); s.M.assign(N * g.Ms, 0); s.A.assign(N * g.Ms, 0);
  s.m.assign(N * g.ms, 0); s.scratch.assign(with_scratch ? N * (size_t)g.scs : 0, 0); s.t.assign(N, 0);
  s.competence = N && nets[0].P.competence ? 1 : 0; s.units = N ? nets[0].units() : 16; s.fanout = N && nets[0].P.fanout ? 1 : 0;
  for (size_t n = 0; n < N; n++) {
    s.t[n] = nets[n].t;
    for (int l = 0; l < 3; l++) {
      const auto& L = nets[n].L[l]; const int N1 = L.N1, D = g.D, U = g.U;
      L.get_hyperplanes_raw(&s.H[n * g.Hs + l * 4L * U * D], &s.Q[n * g.Qs + l * 4L * U]);
      for (int k = 0; k < U; k++) {
        for (int i = 0; i < D; i++) s.p[n * g.ps + (l * (long)U + k) * D + i] = L.gate_weight(k, i);
        for (int c = 0; c < 16; c++) {
          s.m[n * g.ms + (l * (long)U + k) * 16 + c] = L.competence(k, c);
          for (int j = 0; j < N1; j++) s.W[n * g.Ws + g.wofs[l] + (k * 16L + c) * N1 + j] = L.rule(k, c, j);
        }
        for (int j = 0; j < N1; j++) {
          s.M[n * g.Ms + g.mofs[l] + k * N1 + j] = L.active(k, j) ? 1 : 0;
          s.A[n * g.Ms + g.mofs[l] + k * N1 + j] = L.activity(k, j);    // 0 once pruned (dead state)
        }
      }
    }
  }
}
// Max abs difference of the live state of network n against a CPU net (raw hyperplanes and their squared norms, as
// stored, plus the normalised rows the library's hyperplane() returns). Skips the frozen W of pruned columns and,
// after the prune, the activity trace (both are dead state the CPU library no longer keeps).
inline double hg_state_diff(const HgDims& g, const HgHostState& s, size_t n, const hgln::Net<hreal>& net) {
  if (net.units() != g.U || net.d != g.d) return INFINITY;
  double mx = 0;
  auto upd = [&](hreal a, hreal b) { mx = std::max(mx, std::fabs((double)a - (double)b)); };
  for (int l = 0; l < 3; l++) {
    const auto& L = net.L[l]; const int N1 = L.N1, D = g.D, U = g.U;
    for (int k = 0; k < U; k++) {
      for (int b = 0; b < 4; b++) {
        const hreal q = s.Q[n * g.Qs + (l * 4L * U + k * 4 + b)], rs = (hreal)1 / std::sqrt(q);
        upd(q, L.hyperplane_sqnorm(k, b));
        for (int i = 0; i < D; i++) {
          const hreal hv = s.H[n * g.Hs + (l * 4L * U + k * 4 + b) * D + i];
          upd(hv, L.hyperplane_raw(k, b, i)); upd(hv * rs, L.hyperplane(k, b, i));
        }
      }
      for (int i = 0; i < D; i++) upd(s.p[n * g.ps + (l * (long)U + k) * D + i], L.gate_weight(k, i));
      for (int c = 0; c < 16; c++) {
        upd(s.m[n * g.ms + (l * (long)U + k) * 16 + c], L.competence(k, c));
        for (int j = 0; j < N1; j++) if (L.active(k, j)) upd(s.W[n * g.Ws + g.wofs[l] + (k * 16L + c) * N1 + j], L.rule(k, c, j));
      }
      for (int j = 0; j < N1; j++) {
        upd(s.M[n * g.Ms + g.mofs[l] + k * N1 + j], L.active(k, j) ? 1 : 0);
        if (!L.pruned) upd(s.A[n * g.Ms + g.mofs[l] + k * N1 + j], L.activity(k, j));
      }
    }
  }
  return mx;
}
