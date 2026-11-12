// SPDX-License-Identifier: AGPL-3.0-or-later
// HGLN layer kernels for SVE at one fixed vector length. CMake compiles this file once per length FORGE builds a
// table for (-msve-vector-bits=128/256/512/1024); each copy exports hgln::simd::k_sve<B>::table. 1024-bit SVE has
// 32 float lanes, more than a layer's 16, so that length uses the vector-extension kernels for float too.
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
#if defined(__aarch64__)
#if !defined(__ARM_FEATURE_SVE_BITS) || __ARM_FEATURE_SVE_BITS < 128
#error "kernels_sve.cpp must be compiled with SVE and -msve-vector-bits=<128..1024>"
#endif
#define HGLN_CAT2(a, b) a##b
#define HGLN_CAT(a, b) HGLN_CAT2(a, b)
#define HGLN_SIMD_NS HGLN_CAT(k_sve, __ARM_FEATURE_SVE_BITS)
#define HGLN_SIMD_NAME "sve"
#define HGLN_SIMD_TG sve_vl<__ARM_FEATURE_SVE_BITS>
#define HGLN_SIMD_BITS __ARM_FEATURE_SVE_BITS
#ifndef HGLN_SIMD_FORGE_VEC
#define HGLN_SIMD_FORGE_VEC (__ARM_FEATURE_SVE_BITS <= 512)
#endif
#include "kernels.inc"
#endif
