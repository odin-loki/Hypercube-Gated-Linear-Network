// SPDX-License-Identifier: AGPL-3.0-or-later
// HGLN layer kernels for x86-64-v3 (AVX2, FMA, F16C; FMA is not used: -ffp-contract=off). Flags as FORGE's
// simd/src/kernels_avx2.cpp: -mavx2 -mfma -mf16c.
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
#define HGLN_SIMD_NS k_avx2
#define HGLN_SIMD_NAME "avx2"
#define HGLN_SIMD_TG x86_64_v3
#define HGLN_SIMD_BITS 256
#ifndef HGLN_SIMD_FORGE_VEC
#define HGLN_SIMD_FORGE_VEC 1
#endif
#include "kernels.inc"
#if HGLN_SIMD_FORGE_VEC
static_assert(forge::simd::vec<float, forge::simd::x86_64_v3>::K == 1, "avx2 TU must build native AVX2 kernels");
#endif
