// SPDX-License-Identifier: AGPL-3.0-or-later
// kernels.inc - body of one per-target kernel table. Included by src/simd/kernels_<target>.cpp after defining
//   HGLN_SIMD_NS     namespace of this table (unique per TU), e.g. avx512
//   HGLN_SIMD_NAME   table name, the same as FORGE's target name, e.g. "avx512"
//   HGLN_SIMD_TG     FORGE target tag for the float32 kernels (forge::simd::vec<float, Tg>), e.g. x86_64_v4
// CMake compiles each TU with that target's flags plus -ffp-contract=off.
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
//
// One-definition rule: the same inline code is compiled here with different ISA flags, so nothing this TU emits may
// be shared with other TUs by name. The kernels are Layer::gates_k/rules_k/update_k instantiated with a policy type from
// an unnamed namespace (internal linkage); the gate is made internal through HGLN_HD; the float vector wrapper lives
// in this TU's namespace; vector loads/stores are always_inline. FORGE's vec operations are instantiated with the
// same tag and the same flags as FORGE's own kernels_<target>.cpp, so their copies are interchangeable.
#define HGLN_SIMD_KERNEL_TU 1
#define HGLN_HD static inline __attribute__((always_inline))
#include "hgln.hpp"
#include <type_traits>
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
// N float lanes as N/fv::N FORGE Tier P vectors (16 lanes: avx512 one zmm, avx2 two ymm, neon four q registers).
// Only add, sub, mul and div are used: each is one IEEE operation per lane, exactly as in the vector-extension
// kernels, so the bits are the same on every target (FORGE's fma is deliberately not used).
template <class Tg, int NL> struct FN {
  using fv = forge::simd::vec<float, Tg>;
  static constexpr int N = fv::N, R = NL / N;
  static_assert(NL % fv::N == 0, "lanes must be a whole number of FORGE vectors");
  fv r[R];
};
#define HGLN_FN_OP(op, fn)                                                                                           \
  template <class Tg, int NL> HGLN_INLINE FN<Tg, NL> operator op(const FN<Tg, NL>& a, const FN<Tg, NL>& b) {         \
    FN<Tg, NL> o; for (int i = 0; i < FN<Tg, NL>::R; i++) o.r[i] = forge::simd::fn(a.r[i], b.r[i]); return o; }     \
  template <class Tg, int NL> HGLN_INLINE FN<Tg, NL> operator op(const FN<Tg, NL>& a, float s) {                     \
    const auto sv = forge::simd::splat<Tg>(s);                                                                       \
    FN<Tg, NL> o; for (int i = 0; i < FN<Tg, NL>::R; i++) o.r[i] = forge::simd::fn(a.r[i], sv); return o; }         \
  template <class Tg, int NL> HGLN_INLINE FN<Tg, NL> operator op(float s, const FN<Tg, NL>& b) {                     \
    const auto sv = forge::simd::splat<Tg>(s);                                                                       \
    FN<Tg, NL> o; for (int i = 0; i < FN<Tg, NL>::R; i++) o.r[i] = forge::simd::fn(sv, b.r[i]); return o; }         \
  template <class Tg, int NL> HGLN_INLINE FN<Tg, NL>& operator op##=(FN<Tg, NL>& a, const FN<Tg, NL>& b) { a = a op b; return a; }
HGLN_FN_OP(+, add)
HGLN_FN_OP(-, sub)
HGLN_FN_OP(*, mul)
HGLN_FN_OP(/, div)
#undef HGLN_FN_OP
template <class Tg, int NL> struct ForgeL {
  typedef float S; typedef FN<Tg, NL> V;
  static constexpr int N = V::N;
  static HGLN_INLINE V load(const float* p) { V v; for (int i = 0; i < V::R; i++) v.r[i] = forge::simd::load<Tg>(p + i * N); return v; }
  static HGLN_INLINE void store(float* p, const V& v) { for (int i = 0; i < V::R; i++) forge::simd::store<Tg>(p + i * N, v.r[i]); }
  static HGLN_INLINE V zero() { V v; const auto z = forge::simd::splat<Tg>(0.0f); for (int i = 0; i < V::R; i++) v.r[i] = z; return v; }
};
// Policy: FORGE vectors wherever the lane count is a whole number of FORGE vectors (16 corners; 16 or 8 units; on
// avx512 8 units and the 4-lane part of 12 units fall back to the vector extensions, which give the same bits).
template <class Tg> struct ForgeOps : ForgeL<Tg, 16> {
  template <int NL> struct L
      : std::conditional_t<NL % forge::simd::vec<float, Tg>::N == 0, ForgeL<Tg, NL>, typename detail::VxOps<float>::template L<NL>> {};
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

template <class T, class P, int UU> void gat(const Layer<T>& L, const T* u, LayerWork<T>& w) { L.template gates_k<P, UU>(u, w); }
template <class T, class P, int UU> void rul(const Layer<T>& L, const T* in, LayerWork<T>& w, T* xg) { L.template rules_k<P, UU>(in, w, xg); }
template <class T, class P, int UU> void upd(Layer<T>& L, T y, T mu, T glr, const T* es, const T* u, const LayerWork<T>& w) {
  L.template update_k<P, UU>(y, mu, glr, es, u, w);
}
}  // namespace

extern const Table table;
#define HGLN_SIMD_K3(T, P) {{&gat<T, P, 8>, &gat<T, P, 12>, &gat<T, P, 16>}, {&rul<T, P, 8>, &rul<T, P, 12>, &rul<T, P, 16>}, \
                             {&upd<T, P, 8>, &upd<T, P, 12>, &upd<T, P, 16>}}
const Table table = {HGLN_SIMD_NAME, HGLN_SIMD_BITS, HGLN_SIMD_K3(float, FOps), HGLN_SIMD_K3(double, DOps)};
#undef HGLN_SIMD_K3
}  // namespace hgln::simd::HGLN_SIMD_NS
