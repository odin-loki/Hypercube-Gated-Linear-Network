// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_cuda.cu - nvcc (CUDA runtime API) host program for the HGLN CUDA kernel, an alternative to the FORGE build's
// hgln_gpu tool. Compiled with nvcc 12.0 and run on an RTX 3090 (sm_86).
// Build: nvcc -O3 -std=c++17 -arch=sm_86 -Xcompiler -ffp-contract=off -fmad=false -Iinclude -Icuda cuda/hgln_cuda.cu -o hgln_cuda
//        (add -DHG_REAL=float for float32; -fmad=false keeps the library's operation order)
//   hgln_cuda selftest d nets T [v42] [units=N] [fanout] [v5]  compare the GPU result with the CPU network (max abs
//                                      prediction / state difference)
//   hgln_cuda bench d nets T [v42] [units=N] [fanout] [v5]     time T samples for `nets` independent networks
//   v42: the version 4.2 competence-weighted mixture (Params::competence = true); default the 4.3 plain average
//   units=8|12|16: Params::units (default 16); fanout: Params::fanout (layer 3 reads layer 1's outputs)
//   v5: the version 5.0 options (latch 0.1, latch credit 0.5, gate_err 2, glob 0.95, top-6 corners); default 4.3
// The per-sample scratch goes in dynamic shared memory when it fits (d <= 128 in float64 fits in 48 KB for every
// unit count), otherwise in global memory. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#include "hgln_host.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
using hgln::Net;

struct Dev { hreal *H, *Q, *p, *W, *M, *A, *m, *S, *X, *Y, *P; long* t; };
static Dev upload(const HgDims& g, HgHostState& st, const std::vector<hreal>& X, const std::vector<hreal>& Y, int nets, int T) {
  Dev v; v.S = nullptr; (void)g;
  CK(cudaMalloc(&v.H, st.H.size() * sizeof(hreal))); CK(cudaMalloc(&v.Q, st.Q.size() * sizeof(hreal))); CK(cudaMalloc(&v.p, st.p.size() * sizeof(hreal)));
  CK(cudaMalloc(&v.W, st.W.size() * sizeof(hreal))); CK(cudaMalloc(&v.M, st.M.size() * sizeof(hreal)));
  CK(cudaMalloc(&v.A, st.A.size() * sizeof(hreal))); CK(cudaMalloc(&v.m, st.m.size() * sizeof(hreal)));
  if (!st.scratch.empty()) CK(cudaMalloc(&v.S, st.scratch.size() * sizeof(hreal)));
  CK(cudaMalloc(&v.X, X.size() * sizeof(hreal)));
  CK(cudaMalloc(&v.Y, Y.size() * sizeof(hreal))); CK(cudaMalloc(&v.P, (size_t)nets * T * sizeof(hreal))); CK(cudaMalloc(&v.t, nets * sizeof(long)));
  CK(cudaMemcpy(v.H, st.H.data(), st.H.size() * sizeof(hreal), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.Q, st.Q.data(), st.Q.size() * sizeof(hreal), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.p, st.p.data(), st.p.size() * sizeof(hreal), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.W, st.W.data(), st.W.size() * sizeof(hreal), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.M, st.M.data(), st.M.size() * sizeof(hreal), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.A, st.A.data(), st.A.size() * sizeof(hreal), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.m, st.m.data(), st.m.size() * sizeof(hreal), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.X, X.data(), X.size() * sizeof(hreal), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.Y, Y.data(), Y.size() * sizeof(hreal), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(v.t, st.t.data(), nets * sizeof(long), cudaMemcpyHostToDevice));
  return v;
}
int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: hgln_cuda selftest|bench d nets T [v42] [units=N] [fanout] [v5]\n"); return 1; }
  std::string mode = argv[1]; int d = atoi(argv[2]), nets = atoi(argv[3]), T = atoi(argv[4]), units = 16; bool v42 = false, fanout = false, v5 = false;
  for (int i = 5; i < argc; i++) {
    std::string a = argv[i];
    if (a == "v42") v42 = true;
    else if (a == "fanout") fanout = true;
    else if (a == "v5") v5 = true;
    else if (a.rfind("units=", 0) == 0) units = atoi(a.c_str() + 6);
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
  }
  if (d < 1 || d > HG_MAXN || !hg_valid_units(units)) { fprintf(stderr, "need 1 <= d <= %d and units 8, 12 or 16\n", HG_MAXN); return 1; }
  HgDims g = hg_dims(d, units); hgln::Rng rng(12345);
  std::vector<hreal> X((size_t)T * d), Y((size_t)nets * T);
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < d; i++) X[(size_t)t * d + i] = 2 * rng.uni() - 1;
    hreal y = std::sin(3 * X[(size_t)t * d]) * std::cos(2 * X[(size_t)t * d + 1]) + 0.5 * X[(size_t)t * d + 2 % d] * X[(size_t)t * d + 3 % d];
    for (int n = 0; n < nets; n++) Y[(size_t)n * T + t] = n % 3 == 0 ? y : (n % 3 == 1 ? -y : 0.5 * y);
  }
  hgln::Params prm = hgln::Params::v43(d > 16 ? 0.02 : 0.2, d > 16 ? 0.02 : 0.05);   // 4.3 unless options are given
  prm.competence = v42; prm.units = units; prm.fanout = fanout;
  if (v5) { prm.latch = 0.1; prm.latch_credit = 0.5; prm.gate_err = 2; prm.glob = 0.95; prm.corners = 6; }
  std::vector<Net<hreal>> init(nets), ref(nets);
  for (int n = 0; n < nets; n++) { init[n].init(d, 7 + n, prm); ref[n] = init[n]; }
  // scratch in dynamic shared memory if the device allows that much per block, else in global memory
  size_t smem = (size_t)g.scs * sizeof(hreal); int dev = 0, optin = 0; CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  bool use_smem = smem <= (size_t)optin;
  if (use_smem && smem > 48 * 1024) CK(cudaFuncSetAttribute(hgln_run, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  HgHostState st; hg_pack(g, init, st, !use_smem);
  Dev v = upload(g, st, X, Y, nets, T);
  const int threads = 256;
  printf("scratch: %zu bytes per network in %s memory\n", smem, use_smem ? "shared" : "global");
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  hgln_run<<<nets, threads, use_smem ? smem : 0>>>(g, v.H, v.Q, v.p, v.W, v.M, v.A, v.m, v.S, v.X, v.Y, v.P, v.t, T, (hreal)prm.glr0, (hreal)prm.glr1,
                              (hreal)prm.tau, (hreal)prm.glob, (hreal)prm.mu, prm.warm, st.competence, st.units, st.fanout,
                              st.latch, st.credit, st.gate_err, st.corners);
  CK(cudaGetLastError()); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  printf("GPU: %d networks x %d samples in %.2f ms = %.2f us per sample (all networks in parallel)\n", nets, T, ms, 1000.0 * ms / T);
  if (mode == "selftest") {
    std::vector<hreal> P((size_t)nets * T);
    CK(cudaMemcpy(P.data(), v.P, P.size() * sizeof(hreal), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(st.H.data(), v.H, st.H.size() * sizeof(hreal), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(st.Q.data(), v.Q, st.Q.size() * sizeof(hreal), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(st.p.data(), v.p, st.p.size() * sizeof(hreal), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(st.W.data(), v.W, st.W.size() * sizeof(hreal), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(st.M.data(), v.M, st.M.size() * sizeof(hreal), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(st.A.data(), v.A, st.A.size() * sizeof(hreal), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(st.m.data(), v.m, st.m.size() * sizeof(hreal), cudaMemcpyDeviceToHost));
    double wp = 0, ws = 0;
    for (int n = 0; n < nets; n++) {
      for (int t = 0; t < T; t++) { hreal pr = ref[n].predict(&X[(size_t)t * d]); ref[n].learn(Y[(size_t)n * T + t]); wp = std::max(wp, std::fabs((double)pr - (double)P[(size_t)n * T + t])); }
      ws = std::max(ws, hg_state_diff(g, st, n, ref[n]));
    }
    printf("selftest d=%d nets=%d T=%d competence=%d units=%d %s%s: max |prediction diff| = %.3g, max |state diff| = %.3g (%s)\n", d, nets, T,
           st.competence, st.units, st.fanout ? "fanout" : "chain", v5 ? " v5" : "", wp, ws,
           sizeof(hreal) == 4 ? "float32" : "float64");
  }
  hreal* bufs[] = {v.H, v.Q, v.p, v.W, v.M, v.A, v.m, v.S, v.X, v.Y, v.P};
  for (hreal* b : bufs) if (b) CK(cudaFree(b));
  CK(cudaFree(v.t)); CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  return 0;
}
