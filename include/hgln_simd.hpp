// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_simd.hpp - run-time SIMD dispatch for HGLN's per-sample layer kernels (CMake build with FORGE only).
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
//
// The layer kernels (Layer<T>::forward_k / update_k in hgln.hpp) are compiled once per CPU target in
// src/simd/kernels_<target>.cpp, each with that target's flags and -ffp-contract=off. One table of function pointers
// is chosen for the process on first use, from FORGE's selection (forge::simd::active(): the best target this CPU
// and OS support, or FORGE_SIMD_TARGET=scalar|avx2|avx512|neon|sve|rvv if that one is usable here). Every table
// gives bit-identical results, equal to the header-only build (tests/test_simd.cpp checks this).
//
// Included by hgln.hpp when HGLN_WITH_FORGE is defined; link the CMake target hgln_simd. C++17.
#pragma once
#include <atomic>
#include <string>

namespace hgln {
template <class T> struct Layer;
template <class T> struct LayerWork;

namespace simd {
template <class T> struct LayerKernels {
  void (*forward)(const Layer<T>&, const T* in, const T* u, LayerWork<T>& w, T* xg);
  void (*update)(Layer<T>&, T y, T mu, T glr, const T* es, const T* u, const LayerWork<T>& w);
};
struct Table {
  const char* target;                   // "scalar", "avx2", "avx512", "neon", "sve", "rvv"
  int vector_bits;                      // register width the table was built for (0 = baseline ISA)
  LayerKernels<float> f32;
  LayerKernels<double> f64;
};

const char* const* targets();           // null-terminated list of the tables built into this binary
const Table* table(const char* name);   // that table if it is built and this CPU/OS can run it, else nullptr
bool force(const char* name);           // make `name` the active table (false, and no change, if not usable)
void install(const Table* t);           // make any table active (tests); nullptr = back to the automatic choice
std::string cpu_summary();              // FORGE's one-line description of the CPU features
const Table& choose();                  // the automatic choice (FORGE_SIMD_TARGET honoured); also installs it

extern std::atomic<const Table*> g_active;   // current table; null until first use
inline const Table& active() {
  const Table* t = g_active.load(std::memory_order_acquire);
  return t ? *t : choose();
}
template <class T> const LayerKernels<T>& layer_kernels();
template <> inline const LayerKernels<float>& layer_kernels<float>() { return active().f32; }
template <> inline const LayerKernels<double>& layer_kernels<double>() { return active().f64; }
}  // namespace simd
}  // namespace hgln
