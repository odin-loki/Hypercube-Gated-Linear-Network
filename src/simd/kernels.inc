// SPDX-License-Identifier: AGPL-3.0-or-later
// kernels.inc - body of one per-target kernel table. Included by src/simd/kernels_<target>.cpp after defining
//   HGLN_SIMD_NS     namespace of this table (unique per TU), e.g. avx512
//   HGLN_SIMD_NAME   table name, the same as FORGE's target name, e.g. "avx512"
//   HGLN_SIMD_TG     FORGE target tag for the float32 kernels (forge::simd::vec<float, Tg>), e.g. x86_64_v4
// CMake compiles each TU with that target's flags plus -ffp-contract=off.
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
//
// One-definition rule: the same inline code is compiled here with different ISA flags, so nothing this TU emits may
// be shared with other TUs by name. The kernels are Layer::forward_k/update_k instantiated with a policy type from
// an unnamed namespace (internal linkage); the gate is made internal through HGLN_HD; the float vector wrapper lives
// in this TU's namespace; vector loads/stores are always_inline. FORGE's vec operations are instantiated with the
// same tag and the same flags as FORGE's own kernels_<target>.cpp, so their copies are interchangeable.
#define HGLN_SIMD_KERNEL_TU 1
#define HGLN_HD static inline __attribute__((always_inline))
#include "hgln.hpp"
#if HGLN_SIMD_FORGE_VEC
#include <forge/simd/vec.hpp>
#endif

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpsabi"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace hgln::simd::HGLN_SIMD_NS {
#if HGLN_SIMD_FORGE_VEC
// 16 float lanes as 16/N FORGE Tier P vectors (avx512: one zmm; avx2: two ymm; neon: four q registers).
// Only add, sub, mul and div are used: each is one IEEE operation per lane, exactly as in the vector-extension
// kernels, so the bits are the same on every target (FORGE's fma is deliberately not used).
template <class Tg> struct F16 {
  using fv = forge::simd::vec<float, Tg>;
  static constexpr int N = fv::N, R = 16 / N;
  static_assert(16 % fv::N == 0, "16 lanes must be a whole number of FORGE vectors");
  fv r[R];
};
#define HGLN_F16_OP(op, fn)                                                                                          \
  template <class Tg> HGLN_INLINE F16<Tg> operator op(const F16<Tg>& a, const F16<Tg>& b) {                          \
    F16<Tg> o; for (int i = 0; i < F16<Tg>::R; i++) o.r[i] = forge::simd::fn(a.r[i], b.r[i]); return o; }           \
  template <class Tg> HGLN_INLINE F16<Tg> operator op(const F16<Tg>& a, float s) {                                   \
    const auto sv = forge::simd::splat<Tg>(s);                                                                       \
    F16<Tg> o; for (int i = 0; i < F16<Tg>::R; i++) o.r[i] = forge::simd::fn(a.r[i], sv); return o; }               \
  template <class Tg> HGLN_INLINE F16<Tg> operator op(float s, const F16<Tg>& b) {                                   \
    const auto sv = forge::simd::splat<Tg>(s);                                                                       \
    F16<Tg> o; for (int i = 0; i < F16<Tg>::R; i++) o.r[i] = forge::simd::fn(sv, b.r[i]); return o; }               \
  template <class Tg> HGLN_INLINE F16<Tg>& operator op##=(F16<Tg>& a, const F16<Tg>& b) { a = a op b; return a; }
HGLN_F16_OP(+, add)
HGLN_F16_OP(-, sub)
HGLN_F16_OP(*, mul)
HGLN_F16_OP(/, div)
#undef HGLN_F16_OP
template <class Tg> struct ForgeOps {
  typedef F16<Tg> V;
  static constexpr int N = V::N;
  static HGLN_INLINE V load(const float* p) { V v; for (int i = 0; i < V::R; i++) v.r[i] = forge::simd::load<Tg>(p + i * N); return v; }
  static HGLN_INLINE void store(float* p, const V& v) { for (int i = 0; i < V::R; i++) forge::simd::store<Tg>(p + i * N, v.r[i]); }
  static HGLN_INLINE V zero() { V v; const auto z = forge::simd::splat<Tg>(0.0f); for (int i = 0; i < V::R; i++) v.r[i] = z; return v; }
};
#endif

namespace {
// Policies with internal linkage, so Layer<T>::forward_k<P> / update_k<P> are private to this TU.
#if HGLN_SIMD_FORGE_VEC
struct FOps : ForgeOps<forge::simd::HGLN_SIMD_TG> {};
#else
struct FOps : detail::VxOps<float> {};
#endif
struct DOps : detail::VxOps<double> {};

template <class T, class P> void fwd(const Layer<T>& L, const T* in, const T* u, LayerWork<T>& w, T* xg) {
  L.template forward_k<P>(in, u, w, xg);
}
template <class T, class P> void upd(Layer<T>& L, T y, T mu, T glr, const T* es, const T* u, const LayerWork<T>& w) {
  L.template update_k<P>(y, mu, glr, es, u, w);
}
}  // namespace

extern const Table table;
const Table table = {HGLN_SIMD_NAME, HGLN_SIMD_BITS, {&fwd<float, FOps>, &upd<float, FOps>}, {&fwd<double, DOps>, &upd<double, DOps>}};
}  // namespace hgln::simd::HGLN_SIMD_NS
