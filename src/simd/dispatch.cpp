// SPDX-License-Identifier: AGPL-3.0-or-later
// dispatch.cpp - picks HGLN's kernel table for this process from FORGE's CPU detection and target choice.
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
// Built with baseline flags. HGLN_SIMD_HAVE_<NAME> (set by cmake/hgln_simd.cmake) says which tables exist.
#include "hgln_simd.hpp"
#include <forge/simd/dispatch.hpp>
#include <cstring>

namespace hgln::simd {
namespace k_scalar { extern const Table table; }
#if defined(HGLN_SIMD_HAVE_AVX2)
namespace k_avx2 { extern const Table table; }
#endif
#if defined(HGLN_SIMD_HAVE_AVX512)
namespace k_avx512 { extern const Table table; }
#endif
#if defined(HGLN_SIMD_HAVE_NEON)
namespace k_neon { extern const Table table; }
#endif
#if defined(HGLN_SIMD_HAVE_SVE128)
namespace k_sve128 { extern const Table table; }
#endif
#if defined(HGLN_SIMD_HAVE_SVE256)
namespace k_sve256 { extern const Table table; }
#endif
#if defined(HGLN_SIMD_HAVE_SVE512)
namespace k_sve512 { extern const Table table; }
#endif
#if defined(HGLN_SIMD_HAVE_SVE1024)
namespace k_sve1024 { extern const Table table; }
#endif
#if defined(HGLN_SIMD_HAVE_RVV128)
namespace k_rvv128 { extern const Table table; }
#endif
#if defined(HGLN_SIMD_HAVE_RVV256)
namespace k_rvv256 { extern const Table table; }
#endif
#if defined(HGLN_SIMD_HAVE_RVV512)
namespace k_rvv512 { extern const Table table; }
#endif
#if defined(HGLN_SIMD_HAVE_RVV1024)
namespace k_rvv1024 { extern const Table table; }
#endif

std::atomic<const Table*> g_active{nullptr};

namespace {
// Scalable vectors: the table built for exactly this vector length (FORGE's rule), else none.
const Table* sve_for(int bytes) {
  switch (bytes * 8) {
#if defined(HGLN_SIMD_HAVE_SVE128)
    case 128: return &k_sve128::table;
#endif
#if defined(HGLN_SIMD_HAVE_SVE256)
    case 256: return &k_sve256::table;
#endif
#if defined(HGLN_SIMD_HAVE_SVE512)
    case 512: return &k_sve512::table;
#endif
#if defined(HGLN_SIMD_HAVE_SVE1024)
    case 1024: return &k_sve1024::table;
#endif
    default: (void)bytes; return nullptr;
  }
}
const Table* rvv_for(int bytes) {
  switch (bytes * 8) {
#if defined(HGLN_SIMD_HAVE_RVV128)
    case 128: return &k_rvv128::table;
#endif
#if defined(HGLN_SIMD_HAVE_RVV256)
    case 256: return &k_rvv256::table;
#endif
#if defined(HGLN_SIMD_HAVE_RVV512)
    case 512: return &k_rvv512::table;
#endif
#if defined(HGLN_SIMD_HAVE_RVV1024)
    case 1024: return &k_rvv1024::table;
#endif
    default: (void)bytes; return nullptr;
  }
}
// HGLN's table of that name, if built (no usability check).
const Table* built(const char* n) {
  if (!std::strcmp(n, "scalar")) return &k_scalar::table;
#if defined(HGLN_SIMD_HAVE_AVX2)
  if (!std::strcmp(n, "avx2")) return &k_avx2::table;
#endif
#if defined(HGLN_SIMD_HAVE_AVX512)
  if (!std::strcmp(n, "avx512")) return &k_avx512::table;
#endif
#if defined(HGLN_SIMD_HAVE_NEON)
  if (!std::strcmp(n, "neon")) return &k_neon::table;
#endif
  if (!std::strcmp(n, "sve")) return sve_for(forge::simd::detect().sve_bytes);
  if (!std::strcmp(n, "rvv")) return rvv_for(forge::simd::detect().rvv_bytes);
  return nullptr;
}
}  // namespace

const char* const* targets() {
  static const char* const names[] = {
#if defined(HGLN_SIMD_HAVE_AVX512)
    "avx512",
#endif
#if defined(HGLN_SIMD_HAVE_AVX2)
    "avx2",
#endif
#if defined(HGLN_SIMD_HAVE_SVE128) || defined(HGLN_SIMD_HAVE_SVE256) || defined(HGLN_SIMD_HAVE_SVE512) || defined(HGLN_SIMD_HAVE_SVE1024)
    "sve",
#endif
#if defined(HGLN_SIMD_HAVE_RVV128) || defined(HGLN_SIMD_HAVE_RVV256) || defined(HGLN_SIMD_HAVE_RVV512) || defined(HGLN_SIMD_HAVE_RVV1024)
    "rvv",
#endif
#if defined(HGLN_SIMD_HAVE_NEON)
    "neon",
#endif
    "scalar", nullptr};
  return names;
}

// Usable = HGLN built it and FORGE says this CPU/OS can run FORGE's table of the same name (FORGE checks the whole
// feature set, OS register state, and the scalable vector length).
const Table* table(const char* name) {
  if (!name) return nullptr;
  const Table* t = built(name);
  return t && forge::simd::by_name(name) ? t : nullptr;
}

const Table& choose() {
  // FORGE's choice: FORGE_SIMD_TARGET (or FORGE_LANE_TARGET) if usable, else the best usable target.
  const Table* t = table(forge::simd::active().target);
  if (!t)   // FORGE chose a table HGLN did not build (e.g. a vector length HGLN has no table for): best of ours
    for (const char* const* n = targets(); *n && !t; n++) t = table(*n);
  if (!t) t = &k_scalar::table;
  const Table* expected = nullptr;
  g_active.compare_exchange_strong(expected, t, std::memory_order_acq_rel);   // first caller wins; a force() stays
  return *g_active.load(std::memory_order_acquire);
}

bool force(const char* name) {
  const Table* t = table(name);
  if (t) g_active.store(t, std::memory_order_release);
  return t != nullptr;
}

void install(const Table* t) {
  if (t) { g_active.store(t, std::memory_order_release); return; }
  g_active.store(nullptr, std::memory_order_release);
  choose();
}

std::string cpu_summary() { return forge::simd::detect().summary(); }
}  // namespace hgln::simd
