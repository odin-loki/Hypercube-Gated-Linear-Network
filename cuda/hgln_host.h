// hgln_host.h - host helpers: copy a CPU hgln::Net (any training state) into / out of the flat kernel layout.
#pragma once
#include "hgln_kernel.h"
#include "hgln.hpp"
#include <vector>
struct HgHostState { std::vector<hreal> H, p, W, M, A, m, scratch; std::vector<long> t; };
inline void hg_pack(const HgDims& g, const std::vector<hgln::Net<hreal>>& nets, HgHostState& s) {
  int N = (int)nets.size();
  s.H.assign(N * g.Hs, 0); s.p.assign(N * g.ps, 0); s.W.assign(N * g.Ws, 0); s.M.assign(N * g.Ms, 0); s.A.assign(N * g.Ms, 0);
  s.m.assign(N * g.ms, 0); s.scratch.assign((size_t)N * g.scs, 0); s.t.assign(N, 0);
  for (int n = 0; n < N; n++) {
    s.t[n] = nets[n].t;
    for (int l = 0; l < 3; l++) { const auto& L = nets[n].L[l];
      std::copy(L.H.begin(), L.H.end(), s.H.begin() + n * g.Hs + l * 64L * g.D);
      std::copy(L.p.begin(), L.p.end(), s.p.begin() + n * g.ps + l * 16L * g.D);
      std::copy(L.W.begin(), L.W.end(), s.W.begin() + n * g.Ws + g.wofs[l]);
      std::copy(L.M.begin(), L.M.end(), s.M.begin() + n * g.Ms + g.mofs[l]);
      std::copy(L.A.begin(), L.A.end(), s.A.begin() + n * g.Ms + g.mofs[l]);
      std::copy(L.m.begin(), L.m.end(), s.m.begin() + n * g.ms + l * 256L); }
  }
}
// max abs difference of all persistent state of network n against a CPU net
inline double hg_state_diff(const HgDims& g, const HgHostState& s, int n, const hgln::Net<hreal>& net) {
  double mx = 0;
  auto cmp = [&](const std::vector<hreal>& v, long base, const std::vector<hreal>& ref) { for (size_t i = 0; i < ref.size(); i++) mx = std::max(mx, std::fabs((double)v[base + i] - (double)ref[i])); };
  for (int l = 0; l < 3; l++) { const auto& L = net.L[l];
    cmp(s.H, n * g.Hs + l * 64L * g.D, L.H); cmp(s.p, n * g.ps + l * 16L * g.D, L.p); cmp(s.W, n * g.Ws + g.wofs[l], L.W);
    cmp(s.M, n * g.Ms + g.mofs[l], L.M); cmp(s.A, n * g.Ms + g.mofs[l], L.A); cmp(s.m, n * g.ms + l * 256L, L.m); }
  return mx;
}
