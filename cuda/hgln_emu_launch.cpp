// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_emu_launch.cpp - "launches" the real HGLN kernel sources (hgln_kernel.h, hgln_q_kernel.h) on the CPU, from an argument array
// laid out exactly as cuLaunchKernel / hipModuleLaunchKernel receive it (kernelParams: one pointer per kernel
// parameter). Used by the mock CUDA/HIP drivers (tests/mock_hgln.c) so that FORGE's runtime -> driver -> kernel
// path runs without a GPU. Each block is gridDim-many runs of blockDim std::threads with a barrier per
// __syncthreads(); blocks run one after another. Dynamic shared memory is a per-block buffer filled with 0xFF bytes
// (NaN as float64 and as float32), so a read of scratch before it is written shows up as a difference.
// Compile once per real type: -DHG_REAL=double exports hgln_emu_launch_f64 and hglnq_emu_launch_f64 (the integer
// kernel), -DHG_REAL=float the _f32 pair.
// It does NOT test GPU behaviour (memory model, scheduling, the compiled PTX / code object).
// Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#define HGLN_EMULATE
#include <barrier>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
namespace {
thread_local int hg_tid_v = 0, hg_nt_v = 1, hg_bid_v = 0;
thread_local std::barrier<>* hg_bar = nullptr;
thread_local double* hg_smem_v = nullptr;
}
#define HG_KERNEL static
#define HG_DEV static inline
#define HG_HD
#define HG_TID (hg_tid_v)
#define HG_NT (hg_nt_v)
#define HG_BID (hg_bid_v)
#define HG_SYNC() hg_bar->arrive_and_wait()
#define HG_SMEM_DECL
#define HG_SMEM (reinterpret_cast<hreal*>(hg_smem_v))
#include "hgln_kernel.h"
#include "hgln_q_kernel.h"

#define HG_EMU_CAT2(a, b) a##b
#define HG_EMU_CAT(a, b) HG_EMU_CAT2(a, b)
#define HG_EMU_TAG_double f64
#define HG_EMU_TAG_float f32
#define HG_EMU_NAME HG_EMU_CAT(hgln_emu_launch_, HG_EMU_CAT(HG_EMU_TAG_, HG_REAL))

// Returns 0, or 1 (invalid launch: block size, or shared scratch requested with too little dynamic shared memory).
extern "C" int HG_EMU_NAME(unsigned grid, unsigned block, unsigned smem_bytes, void** args) {
  if (block < 1 || block > 1024) return 1;
  const HgDims g = *static_cast<const HgDims*>(args[0]);
  auto ptr = [&](int i) { return reinterpret_cast<hreal*>(static_cast<uintptr_t>(*static_cast<const uint64_t*>(args[i]))); };
  // hgln_run(g, H, Q, p, W, M, A, m, scratch, X, Y, P, tcount, T, glr0, glr1, tau, glob, mu, warm, competence, units, fanout,
  //          latch, credit, gate_err, corners)
  hreal *H = ptr(1), *Q = ptr(2), *p = ptr(3), *W = ptr(4), *M = ptr(5), *A = ptr(6), *m = ptr(7), *S = ptr(8);
  const hreal *X = ptr(9), *Y = ptr(10); hreal* P = ptr(11);
  long* tc = reinterpret_cast<long*>(static_cast<uintptr_t>(*static_cast<const uint64_t*>(args[12])));
  const int T = *static_cast<const int*>(args[13]);
  const hreal glr0 = *static_cast<const hreal*>(args[14]), glr1 = *static_cast<const hreal*>(args[15]),
              tau = *static_cast<const hreal*>(args[16]), glob = *static_cast<const hreal*>(args[17]),
              mu = *static_cast<const hreal*>(args[18]);
  const int warm = *static_cast<const int*>(args[19]), competence = *static_cast<const int*>(args[20]),
            units = *static_cast<const int*>(args[21]), fanout = *static_cast<const int*>(args[22]);
  const hreal latch = *static_cast<const hreal*>(args[23]), credit = *static_cast<const hreal*>(args[24]);
  const int gate_err = *static_cast<const int*>(args[25]), corners = *static_cast<const int*>(args[26]);
  if (units != g.U || !hg_valid_units(units)) return 1;     // layout and kernel argument disagree
  if (!S && (size_t)smem_bytes < (size_t)g.scs * sizeof(hreal)) return 1;   // the kernel would run past the buffer
  std::vector<double> smem((smem_bytes + sizeof(double) - 1) / sizeof(double));
  for (unsigned b = 0; b < grid; b++) {
    std::memset(smem.data(), 0xFF, smem.size() * sizeof(double));
    std::barrier<> bar((std::ptrdiff_t)block);
    std::vector<std::thread> th;
    th.reserve(block);
    for (unsigned k = 0; k < block; k++) th.emplace_back([&, k] {
      hg_tid_v = (int)k; hg_nt_v = (int)block; hg_bid_v = (int)b; hg_bar = &bar; hg_smem_v = smem.data();
      hgln_run(g, H, Q, p, W, M, A, m, S, X, Y, P, tc, T, glr0, glr1, tau, glob, mu, warm, competence, units, fanout, latch, credit, gate_err, corners);
    });
    for (auto& t : th) t.join();
  }
  return 0;
}

#define HG_EMUQ_NAME HG_EMU_CAT(hglnq_emu_launch_, HG_EMU_CAT(HG_EMU_TAG_, HG_REAL))
// The integer kernel: hglnq_run(g, C, H, p, W, M, A, scratch, lut, expneg, X, Y, P, tcount, T); no shared memory.
extern "C" int HG_EMUQ_NAME(unsigned grid, unsigned block, unsigned smem_bytes, void** args) {
  (void)smem_bytes;
  if (block < 1 || block > 1024) return 1;
  const HgQDims g = *static_cast<const HgQDims*>(args[0]); const HgQCfg C = *static_cast<const HgQCfg*>(args[1]);
  auto ptr = [&](int i) { return reinterpret_cast<void*>(static_cast<uintptr_t>(*static_cast<const uint64_t*>(args[i]))); };
  int *H = (int*)ptr(2), *p = (int*)ptr(3), *W = (int*)ptr(4), *M = (int*)ptr(5), *A = (int*)ptr(6);
  long long* S = (long long*)ptr(7); const short* lut = (const short*)ptr(8); const long long* ex = (const long long*)ptr(9);
  const hreal *X = (const hreal*)ptr(10), *Y = (const hreal*)ptr(11); hreal* P = (hreal*)ptr(12); long* tc = (long*)ptr(13);
  const int T = *static_cast<const int*>(args[14]);
  if (C.units != g.U || !hg_valid_units(g.U) || !S) return 1;
  for (unsigned b = 0; b < grid; b++) {
    std::barrier<> bar((std::ptrdiff_t)block);
    std::vector<std::thread> th;
    th.reserve(block);
    for (unsigned k = 0; k < block; k++) th.emplace_back([&, k] {
      hg_tid_v = (int)k; hg_nt_v = (int)block; hg_bid_v = (int)b; hg_bar = &bar; hg_smem_v = nullptr;
      hglnq_run(g, C, H, p, W, M, A, S, lut, ex, X, Y, P, tc, T);
    });
    for (auto& t : th) t.join();
  }
  return 0;
}
