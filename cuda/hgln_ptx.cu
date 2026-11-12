// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_ptx.cu - builds the HGLN kernel (hgln_kernel.h) into PTX with plain clang: no CUDA SDK, no nvcc, no headers.
// Same recipe as FORGE's GPU images (github.com/odin-loki/FORGE, gpu/ and ops/gen/gpu_gen.py):
//   clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_70 -O3 -S cuda/hgln_ptx.cu -o hgln.sm70.ptx
// The PTX is self-contained (no libdevice calls) so a CUDA driver can JIT and load it directly (cuModuleLoadData),
// e.g. through FORGE's gpu::cuda_hip runtime or any driver-API host. Kernel symbol: hgln_run (extern "C").
// Add -DHG_REAL=float for the float32 kernel. Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#if !defined(__NVPTX__)
#error "compile with clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_XX"
#endif
#define HGLN_NO_STD_HEADERS
#define HGLN_EMULATE   // keep hgln_kernel.h from defining the nvcc macros; we define them header-free below
#define HG_KERNEL extern "C" __attribute__((global))
#define HG_DEV static __attribute__((device)) inline
#define HG_HD __attribute__((host, device))
#define HGLN_HD static __attribute__((host, device)) inline
#define HG_TID ((int)__nvvm_read_ptx_sreg_tid_x())
#define HG_NT ((int)__nvvm_read_ptx_sreg_ntid_x())
#define HG_BID ((int)__nvvm_read_ptx_sreg_ctaid_x())
#define HG_SYNC() __syncthreads()
#define HG_SMEM_DECL
extern __attribute__((shared)) double hg_smem_raw[];
#define HG_SMEM (reinterpret_cast<hreal*>(hg_smem_raw))
// Device maths without libdevice. sqrt/abs/rint map to single PTX instructions (sqrt.rn, abs, cvt.rni).
static __attribute__((device)) inline double hg_dev_exp(double x) {   // exp(x) to ~1 ulp; used only for x <= 0
  if (x < -708.0) return 0.0;
  if (x > 709.0) x = 709.0;
  const double k = __builtin_rint(x * 1.4426950408889634);
  const double r = (x - k * 6.93147180369123816490e-01) - k * 1.90821492927058770002e-10;
  double e = 1.0 / 6227020800.0;
  const double c[13] = {1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0, 1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0,
                        1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0};
  for (int i = 0; i < 13; i++) e = e * r + c[i];
  const long long bits = (long long)(k + 1023.0) << 52;              // 2^k, exact for -1022 <= k <= 1023
  return e * __builtin_bit_cast(double, bits);
}
#define HG_EXP(x) ((hreal)hg_dev_exp((double)(x)))
#define HG_SQRT(x) __builtin_sqrt(x)
#define HG_FABS(x) __builtin_fabs(x)
#define HG_RINT(x) __builtin_rint(x)
#include "hgln_kernel.h"
