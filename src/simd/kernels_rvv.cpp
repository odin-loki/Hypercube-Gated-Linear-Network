// SPDX-License-Identifier: AGPL-3.0-or-later
// HGLN layer kernels for RVV at one fixed VLEN. CMake compiles this file once per length FORGE builds a table for
// (-mrvv-vector-bits=128/256/512/1024); each copy exports hgln::simd::k_rvv<B>::table. VLEN 1024 has 32 float lanes,
// more than a layer's 16, so that length uses the vector-extension kernels for float too.
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
#if defined(__riscv)
#if !defined(__riscv_v_fixed_vlen)
#error "kernels_rvv.cpp must be compiled with V and -mrvv-vector-bits=<128..1024>"
#endif
#define HGLN_CAT2(a, b) a##b
#define HGLN_CAT(a, b) HGLN_CAT2(a, b)
#define HGLN_SIMD_NS HGLN_CAT(k_rvv, __riscv_v_fixed_vlen)
#define HGLN_SIMD_NAME "rvv"
#define HGLN_SIMD_TG rvv_vl<__riscv_v_fixed_vlen>
#define HGLN_SIMD_BITS __riscv_v_fixed_vlen
#ifndef HGLN_SIMD_FORGE_VEC
#define HGLN_SIMD_FORGE_VEC (__riscv_v_fixed_vlen <= 512)
#endif
#include "kernels.inc"
#endif
