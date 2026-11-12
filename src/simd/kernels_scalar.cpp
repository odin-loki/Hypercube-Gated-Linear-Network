// SPDX-License-Identifier: AGPL-3.0-or-later
// HGLN layer kernels for the baseline ISA of the build (x86-64: SSE2; aarch64: NEON; ...). Always built.
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
#define HGLN_SIMD_NS k_scalar
#define HGLN_SIMD_NAME "scalar"
#define HGLN_SIMD_TG scalar_t
#define HGLN_SIMD_BITS 0
// FORGE's scalar target is 4-wide emulation; on the baseline ISA the vector extensions (lowered to SSE2 / NEON)
// measured about 2x faster for float32, with the same bits, so this table uses them for float32 too.
#ifndef HGLN_SIMD_FORGE_VEC
#define HGLN_SIMD_FORGE_VEC 0
#endif
#include "kernels.inc"
