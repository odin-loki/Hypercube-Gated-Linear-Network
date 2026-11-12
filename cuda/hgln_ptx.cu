// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_ptx.cu - builds the HGLN kernel (hgln_kernel.h) for a GPU with plain clang: no CUDA SDK, no ROCm, no nvcc,
// no headers, no device libraries. Same recipe as FORGE's GPU images (github.com/odin-loki/FORGE, gpu/ and
// ops/gen/gpu_gen.py):
//   CUDA (PTX): clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_70 -O3 -ffp-contract=off
//               -S cuda/hgln_ptx.cu -o hgln.sm70.ptx
//   HIP (AMDGPU code object): clang -x hip --cuda-device-only --no-gpu-bundle-output -nogpuinc -nogpulib
//               --offload-arch=gfx90a -O3 -ffp-contract=off --ld-path=$(which ld.lld) cuda/hgln_ptx.cu -o hgln.gfx90a.hsaco
// The output is self-contained (no libdevice / ocml calls), so a driver can load it directly (cuModuleLoadData /
// hipModuleLoadData), e.g. through FORGE's gpu::cuda_hip runtime (include/hgln_gpu.hpp). Kernel symbols: hgln_run (the
// float network) and hglnq_run (the integer network, cuda/hgln_q_kernel.h), extern "C". Add -DHG_REAL=float for the float32 kernel. -ffp-contract=off keeps every multiply and add separate,
// as in the C++ library, so the GPU does the same arithmetic. Each image also holds an empty marker kernel,
// hgln_real_f64 or hgln_real_f32, so a host can check the image's real type by looking the symbol up.
// Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#if !defined(__NVPTX__) && !defined(__AMDGPU__)
#error "compile with clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_XX, or -x hip --cuda-device-only --offload-arch=gfxNNN"
#endif
#define HGLN_NO_STD_HEADERS
#define HGLN_EMULATE   // keep hgln_kernel.h from defining the nvcc macros; we define them header-free below
#define HG_KERNEL extern "C" __attribute__((global))
#define HG_DEV static __attribute__((device)) inline
#define HG_HD __attribute__((host, device))
#define HGLN_HD static __attribute__((host, device)) inline
#if defined(__NVPTX__)
#define HG_TID ((int)__nvvm_read_ptx_sreg_tid_x())
#define HG_NT ((int)__nvvm_read_ptx_sreg_ntid_x())
#define HG_BID ((int)__nvvm_read_ptx_sreg_ctaid_x())
#define HG_SYNC() __syncthreads()
#else   // AMDGPU: the builtins HIP's own headers use; the barrier is __syncthreads() as HIP defines it
#define HG_TID ((int)__builtin_amdgcn_workitem_id_x())
#define HG_NT ((int)__builtin_amdgcn_workgroup_size_x())
#define HG_BID ((int)__builtin_amdgcn_workgroup_id_x())
#define HG_SYNC() (__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"), __builtin_amdgcn_s_barrier(), \
                   __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"))
#endif
#define HG_SMEM_DECL
extern __attribute__((shared)) double hg_smem_raw[];   // dynamic shared memory (CUDA) / LDS (HIP)
#define HG_SMEM (reinterpret_cast<hreal*>(hg_smem_raw))
// Device maths without libdevice / ocml. sqrt/abs/rint map to native instructions or inline expansions
// (NVPTX: sqrt.rn, abs, cvt.rni; AMDGPU: the correctly rounded sqrt expansion, v_rndne).
[[maybe_unused]] static __attribute__((device)) inline double hg_dev_exp(double x) {   // exp(x) to ~1 ulp (HG_EXP; the kernels now use hgln_exp)
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
[[maybe_unused]] static __attribute__((device)) inline double hg_dev_sqrt(double x) { return __builtin_sqrt(x); }
[[maybe_unused]] static __attribute__((device)) inline float hg_dev_sqrt(float x) { return __builtin_sqrtf(x); }
[[maybe_unused]] static __attribute__((device)) inline double hg_dev_fabs(double x) { return __builtin_fabs(x); }
[[maybe_unused]] static __attribute__((device)) inline float hg_dev_fabs(float x) { return __builtin_fabsf(x); }
#define HG_EXP(x) ((hreal)hg_dev_exp((double)(x)))
#define HG_SQRT(x) hg_dev_sqrt(x)
#define HG_FABS(x) hg_dev_fabs(x)
#define HG_RINT(x) __builtin_rint(x)
#include "hgln_kernel.h"
#include "hgln_q_kernel.h"
#define HG_CAT2(a, b) a##b
#define HG_CAT(a, b) HG_CAT2(a, b)
#define HG_REAL_TAG_double f64
#define HG_REAL_TAG_float f32
extern "C" __attribute__((global)) void HG_CAT(hgln_real_, HG_CAT(HG_REAL_TAG_, HG_REAL))() {}   // hgln_real_f64 / _f32
