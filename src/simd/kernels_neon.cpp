// SPDX-License-Identifier: AGPL-3.0-or-later
// HGLN layer kernels for AArch64 NEON (baseline flags, as FORGE's simd/src/kernels_neon.cpp).
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
#if defined(__aarch64__)
#define HGLN_SIMD_NS k_neon
#define HGLN_SIMD_NAME "neon"
#define HGLN_SIMD_TG apple_m
#define HGLN_SIMD_BITS 128
#ifndef HGLN_SIMD_FORGE_VEC
#define HGLN_SIMD_FORGE_VEC 1
#endif
#include "kernels.inc"
#if HGLN_SIMD_FORGE_VEC
static_assert(forge::simd::vec<float, forge::simd::apple_m>::K == 3, "neon TU must build native NEON kernels");
#endif
#endif
