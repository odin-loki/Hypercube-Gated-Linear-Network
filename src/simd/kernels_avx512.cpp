// SPDX-License-Identifier: AGPL-3.0-or-later
// HGLN layer kernels for x86-64-v4 (AVX-512 F/BW/DQ/VL). Flags as FORGE's simd/src/kernels_avx512.cpp:
// -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma (FMA is not used: -ffp-contract=off).
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
#define HGLN_SIMD_NS k_avx512
#define HGLN_SIMD_NAME "avx512"
#define HGLN_SIMD_TG x86_64_v4
#define HGLN_SIMD_BITS 512
#ifndef HGLN_SIMD_FORGE_VEC
#define HGLN_SIMD_FORGE_VEC 1
#endif
#include "kernels.inc"
#if HGLN_SIMD_FORGE_VEC
static_assert(forge::simd::vec<float, forge::simd::x86_64_v4>::K == 2, "avx512 TU must build native AVX-512 kernels");
#endif
