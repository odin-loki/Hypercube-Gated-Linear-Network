// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_q.hpp - HGLN version 5.0 integer datapath: hgln::NetQ<R>, the same public API as hgln::Net<T> (predict /
// learn / infer with a caller-owned Work, Params, Classifier, run_streams) on int16 state and activations. The forward
// pass and the per-unit control scalars are integer arithmetic; the per-element learning steps are mixed float32 on the
// int16 state (spec section 6), which keeps every CPU target and GPU bit-identical.
// Header-only C++17 (plus, with HGLN_WITH_FORGE, the run-time dispatched kernel tables of src/simd/).
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
//
// The arithmetic is the integer specification in hgln_qspec.hpp, exactly: every table (scalar, avx2, avx512, the
// header-only build) and the research reference (research/alg/hgln_var.hpp, intspec=1) give the same bits
// (tests/test_q.cpp, tests/test_simd.cpp). Layout, built for SIMD:
//   * unit-lane arrays (gates, heatmap, per-unit control scalars): [x][lane], lane = unit, UL = 8 lanes for 8 units and
//     16 for 12 or 16 units (12 units: 4 padding lanes, kept at zero: H = p = 0 there);
//   * H is [i][b][lane] int16, p is [i][lane] int16;
//   * W is [unit][slot pair][corner][slot parity] int16: the 16 rule dot products of one unit are one 16 x int32
//     accumulator over column pairs (AVX-512 VNNI vpdpwssd / AVX-512BW and AVX2 vpmaddwd);
//   * after the prune the inactive columns are removed from W and the activity trace A is freed.
#pragma once
#include "hgln.hpp"
#include "hgln_qspec.hpp"
#include <type_traits>
#include <utility>
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(__AVX512F__))
#include <immintrin.h>
#endif
// Lane vectors are only passed between inline functions of this header (within one kernel TU): -Wpsabi does not apply.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpsabi"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace hgln {

// ---- the model of one layer
struct LayerQ {
  int U = 8, UL = 8;                                          // units; lanes (8, or 16 for 12/16 units)
  int n = 0, D = 0, N1 = 0, nt = 0;                           // inputs, side information (d+1), inputs+bias, prune target
  int cs = 0, cp = 0;                                         // column stride (max active columns) and slot pairs per unit
  bool pruned = false;
  detail::ABuf<int16_t> H;                                    // D x 4 x UL     [i][b][lane]   Q(WB-2), unit length
  detail::ABuf<int16_t> p;                                    // D x UL         [i][lane]      Q(WB-3)
  detail::ABuf<int16_t> W;                                    // U x cp x 32    [unit][pair][corner][parity]  Q(WB-4)
  detail::ABuf<int32_t> A;                                    // U x N1         activity, Q(AB+6), until the prune
  detail::ABuf<int32_t> cols, ncol;                           // U x cs active columns (ascending, bias last); U counts
  size_t widx(int k, int q, int c) const { return ((size_t)k * cp + (q >> 1)) * 32 + (size_t)c * 2 + (q & 1); }
  int slot(int k, int j) const { for (int q = 0; q < ncol[k]; q++) if (cols[(size_t)k * cs + q] == j) return q; return -1; }
  // accessors in the reference (dense) layout, as stored integers
  int32_t hyperplane(int k, int b, int i) const { return H[((size_t)i * 4 + b) * UL + k]; }
  int32_t gate_weight(int k, int i) const { return p[(size_t)i * UL + k]; }
  bool active(int k, int j) const { return slot(k, j) >= 0; }
  int32_t rule(int k, int c, int j) const { int q = slot(k, j); return q < 0 ? 0 : W[widx(k, q, c)]; }
  int32_t activity(int k, int j) const { return pruned ? 0 : A[(size_t)k * N1 + j]; }
  size_t model_bytes() const { return (H.size() + p.size() + W.size()) * 2 + (A.size() + cols.size() + ncol.size()) * 4; }
};

// ---- per-sample scratch of one layer (written by the gates and rules, read by the update)
struct LayerQWork {
  detail::ABuf<int32_t> x;                                    // N1 layer inputs, bias last (Q(AB-2))
  detail::ABuf<int32_t> a, kap, hold;                         // 4 x UL gates (Q(AB-2)); d a / d(12z) and hold (Q14)
  detail::ABuf<int32_t> hs, h, f, sel;                        // 16 x UL soft heatmap, renormalised top-k heatmap, rules
  detail::ABuf<int32_t> o;                                    // UL unit outputs
  detail::ABuf<int64_t> x2;                                   // UL squared norm of the active inputs
  detail::ABuf<uint32_t> xp;                                  // packed input pairs of one unit (scratch)
  void init(int N1, int UL) {
    x.assign(N1, 0); a.assign(4 * UL, 0); kap.assign(4 * UL, 0); hold.assign(4 * UL, 0);
    hs.assign(16 * UL, 0); h.assign(16 * UL, 0); f.assign(16 * UL, 0); sel.assign(16 * UL, 0);
    o.assign(UL, 0); x2.assign(UL, 0); xp.assign((N1 + 1) / 2 + 1, 0);
  }
};
struct WorkQ {
  detail::ABuf<int32_t> s;                                    // side information, d+1, Q(AB-2)
  LayerQWork lw[3];
  detail::ABuf<int32_t> o;                                    // the 3U unit outputs
  int32_t S = 0;                                              // their sum
  template <class NetT> void init(const NetT& net) {
    s.assign(net.d + 1, 0); o.assign(3 * net.L[0].U, 0); S = 0;
    for (int l = 0; l < 3; l++) lw[l].init(net.L[l].N1, net.L[l].UL);
  }
};
// constants of a network, and of one learning step of one layer
struct QCfg { qspec::Fmt F; int K = 8; bool latch = false, credit = false; int32_t th14 = 0; int64_t cscale = 0; };
struct QStep {
  int64_t Mq = 0; uint32_t layer = 0;
  const int32_t* es = nullptr;                                // U rule errors (Q(AB-2))
  const int32_t* ge = nullptr;                                // U gate errors times the gate rate (Q24)
  const int32_t* acc_in = nullptr;                            // 4 x UL gate steps routed here by upper layers (or null)
  int32_t* acc_out = nullptr;                                 // 4 x UL routed to the lower layer (latch credit), or null
  int32_t* scratch = nullptr;                                 // D x 4 x UL
};

namespace detail {
namespace qk {
namespace qs = ::hgln::qspec;
static constexpr int32_t kCB = 53687091;                      // 0.05 at Q30

// ---- small integer lane vectors: GCC/Clang vector extensions (the same types as hgln.hpp's float kernels), otherwise
// a plain array with the same element-wise semantics. Only exact integer operations are used, so every lowering
// (SSE2, AVX2, AVX-512, NEON, scalar) gives the same bits.
template <class T> struct SInt;
template <> struct SInt<int16_t> { typedef int16_t t; };
template <> struct SInt<int32_t> { typedef int32_t t; };
template <> struct SInt<uint32_t> { typedef int32_t t; };
template <> struct SInt<int64_t> { typedef int64_t t; };
template <> struct SInt<uint64_t> { typedef int64_t t; };
#if (defined(__GNUC__) || defined(__clang__)) && !defined(HGLN_NO_VECTOR_EXT)
template <class T, int N> using IV = typename ::hgln::detail::VecT<T, N>::type;
template <class To, class From> HGLN_INLINE To cvt(From v) { return __builtin_convertvector(v, To); }
template <class T, int N> HGLN_INLINE IV<T, N> splat(T s) { IV<T, N> r = {}; return r + s; }
#else
template <class T, int N> struct IV {
  T v[N];
  T& operator[](int i) { return v[i]; }
  const T& operator[](int i) const { return v[i]; }
};
template <class T> struct SNd { typedef T t; };          // non-deduced context for the scalar operand
#define HGLN_QIV_OP(op)                                                                                                    \
  template <class T, int N> inline IV<T, N> operator op(const IV<T, N>& a, const IV<T, N>& b) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = (T)(a.v[i] op b.v[i]); return r; } \
  /* the scalar is not deduced (like the vector extension, v * 12 broadcasts an int into int64 lanes) */                 \
  template <class T, int N> inline IV<T, N> operator op(const IV<T, N>& a, typename SNd<T>::t s) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = (T)(a.v[i] op s); return r; } \
  template <class T, int N> inline IV<T, N> operator op(typename SNd<T>::t s, const IV<T, N>& a) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = (T)(s op a.v[i]); return r; } \
  template <class T, int N> inline IV<T, N>& operator op##=(IV<T, N>& a, const IV<T, N>& b) { a = a op b; return a; }
HGLN_QIV_OP(+) HGLN_QIV_OP(-) HGLN_QIV_OP(*) HGLN_QIV_OP(&) HGLN_QIV_OP(|) HGLN_QIV_OP(^)
#undef HGLN_QIV_OP
#define HGLN_QIV_CMP(op)                                                                                                   \
  template <class T, int N> inline IV<typename SInt<T>::t, N> operator op(const IV<T, N>& a, const IV<T, N>& b) {          \
    IV<typename SInt<T>::t, N> r; for (int i = 0; i < N; i++) r.v[i] = a.v[i] op b.v[i] ? -1 : 0; return r; }
HGLN_QIV_CMP(<) HGLN_QIV_CMP(>) HGLN_QIV_CMP(==)
#undef HGLN_QIV_CMP
template <class T, int N> inline IV<T, N> operator>>(const IV<T, N>& a, int k) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = (T)(a.v[i] >> k); return r; }
template <class T, int N> inline IV<T, N> operator<<(const IV<T, N>& a, int k) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = (T)((typename std::make_unsigned<T>::type)a.v[i] << k); return r; }
template <class T, int N> inline IV<T, N> operator~(const IV<T, N>& a) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = (T)~a.v[i]; return r; }
template <class T, int N> inline IV<T, N> operator-(const IV<T, N>& a) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = (T)-a.v[i]; return r; }
template <class To, class From> inline To cvt(const From& v) { To r; for (int i = 0; i < (int)(sizeof(r.v) / sizeof(r.v[0])); i++) r.v[i] = (decltype(r.v[0] + 0))v.v[i]; return r; }
template <class T, int N> inline IV<T, N> splat(T s) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = s; return r; }
#endif
template <class T, int N> HGLN_INLINE IV<T, N> ld(const T* p) { IV<T, N> r; std::memcpy(&r, p, sizeof r); return r; }
template <class T, int N> HGLN_INLINE void st(T* p, IV<T, N> v) { std::memcpy(p, &v, sizeof v); }
template <class T, int N> HGLN_INLINE IV<T, N> vsel(IV<T, N> m, IV<T, N> a, IV<T, N> b) { return (a & m) | (b & ~m); }   // m: -1 / 0
template <class T, int N> HGLN_INLINE IV<T, N> vclamp(IV<T, N> v, T lo, T hi) {
  const IV<T, N> L = splat<T, N>(lo), H = splat<T, N>(hi);
#if (defined(__GNUC__) || defined(__clang__)) && !defined(HGLN_NO_VECTOR_EXT)
  const IV<T, N> r = v < L ? L : v; return r > H ? H : r;        // vector conditional: min / max instructions
#else
  IV<T, N> r = vsel<T, N>(v < L, L, v); return vsel<T, N>(r > H, H, r);
#endif
}
template <class T, int N> HGLN_INLINE IV<T, N> vrne(IV<T, N> v, int k) {     // spec rne, k >= 1 (no overflow by the callers' bounds)
  return (v + (T)(((T)1 << (k - 1)) - 1) + ((v >> k) & (T)1)) >> k;
}

// 16 rule dot products of one unit over np column pairs: acc[c] = sum W[c] x (int32, wrap-around).
#if defined(__AVX512BW__) && !defined(__AVX512VNNI__) && (defined(__GNUC__) || defined(__clang__))
#define HGLN_Q_VNNI_RT 1
__attribute__((target("avx512f,avx512bw,avx512vnni"))) static void dot16_vnni(const int16_t* Wk, const uint32_t* xp, int np, int32_t* acc) {
  __m512i a = _mm512_setzero_si512();
  for (int q = 0; q < np; q++) a = _mm512_dpwssd_epi32(a, _mm512_load_si512((const void*)(Wk + q * 32)), _mm512_set1_epi32((int)xp[q]));
  _mm512_storeu_si512((void*)acc, a);
}
static inline bool have_vnni() { static const bool v = __builtin_cpu_supports("avx512vnni"); return v; }
#endif
template <class P> HGLN_INLINE void dot16(const int16_t* HGLN_RESTRICT Wk, const uint32_t* HGLN_RESTRICT xp, int np, int32_t* HGLN_RESTRICT acc) {
#if defined(__AVX512BW__)
#if defined(HGLN_Q_VNNI_RT)
  if (have_vnni()) { dot16_vnni(Wk, xp, np, acc); return; }
#endif
  __m512i a = _mm512_setzero_si512();
  for (int q = 0; q < np; q++) {
    const __m512i w = _mm512_load_si512((const void*)(Wk + q * 32)), xv = _mm512_set1_epi32((int)xp[q]);
#if defined(__AVX512VNNI__)
    a = _mm512_dpwssd_epi32(a, w, xv);
#else
    a = _mm512_add_epi32(a, _mm512_madd_epi16(w, xv));
#endif
  }
  _mm512_storeu_si512((void*)acc, a);
#elif defined(__AVX2__)
  __m256i a0 = _mm256_setzero_si256(), a1 = a0;
  for (int q = 0; q < np; q++) {
    const __m256i xv = _mm256_set1_epi32((int)xp[q]);
    a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_load_si256((const __m256i*)(Wk + q * 32)), xv));
    a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_load_si256((const __m256i*)(Wk + q * 32 + 16)), xv));
  }
  _mm256_storeu_si256((__m256i*)acc, a0); _mm256_storeu_si256((__m256i*)(acc + 8), a1);
#else
  uint32_t s[16] = {0};
  for (int q = 0; q < np; q++) {
    const int32_t x0 = (int16_t)(xp[q] & 0xFFFF), x1 = (int16_t)(xp[q] >> 16);
    const int16_t* w = Wk + q * 32;
    for (int c = 0; c < 16; c++) s[c] += (uint32_t)(w[2 * c] * x0 + w[2 * c + 1] * x1);   // a pair cannot overflow (|.| < 2^31)
  }
  for (int c = 0; c < 16; c++) acc[c] = (int32_t)s[c];
#endif
}

// Batcher's odd-even merge sorting network for 16 keys (63 compare-exchanges), generated at compile time.
struct Batcher16 { int n = 0; int8_t a[64] = {}, b[64] = {}; };
constexpr Batcher16 make_batcher16() {
  Batcher16 t;
  for (int p = 1; p < 16; p <<= 1)
    for (int k = p; k >= 1; k >>= 1)
      for (int j = k % p; j + k < 16; j += 2 * k)
        for (int i = 0; i < k && i + j + k < 16; i++)
          if ((i + j) / (2 * p) == (i + j + k) / (2 * p)) { t.a[t.n] = (int8_t)(i + j); t.b[t.n] = (int8_t)(i + j + k); t.n++; }
  return t;
}
static constexpr Batcher16 kBatcher16 = make_batcher16();
static_assert(kBatcher16.n == 63, "Batcher odd-even merge sort of 16 keys has 63 compare-exchanges");
template <int UL> HGLN_INLINE IV<int32_t, UL> vmin32(IV<int32_t, UL> a, IV<int32_t, UL> b) {
#if defined(__AVX512F__) && !defined(HGLN_NO_VECTOR_EXT)
  if constexpr (UL == 16) return (IV<int32_t, UL>)_mm512_min_epi32((__m512i)a, (__m512i)b);
#endif
#if defined(__AVX2__) && !defined(HGLN_NO_VECTOR_EXT)
  if constexpr (UL == 8) return (IV<int32_t, UL>)_mm256_min_epi32((__m256i)a, (__m256i)b);
#endif
  return vsel<int32_t, UL>(a < b, a, b);
}
template <int UL> HGLN_INLINE IV<int32_t, UL> vmax32(IV<int32_t, UL> a, IV<int32_t, UL> b) {
#if defined(__AVX512F__) && !defined(HGLN_NO_VECTOR_EXT)
  if constexpr (UL == 16) return (IV<int32_t, UL>)_mm512_max_epi32((__m512i)a, (__m512i)b);
#endif
#if defined(__AVX2__) && !defined(HGLN_NO_VECTOR_EXT)
  if constexpr (UL == 8) return (IV<int32_t, UL>)_mm256_max_epi32((__m256i)a, (__m256i)b);
#endif
  return vsel<int32_t, UL>(a < b, b, a);
}
// the whole network with compile-time indices (the 16 keys stay in registers)
template <int UL, std::size_t... Q> HGLN_INLINE void batcher16(IV<int32_t, UL>* s, std::index_sequence<Q...>) {
  ((void)([&] { constexpr int i = kBatcher16.a[Q], j = kBatcher16.b[Q];
                const IV<int32_t, UL> lo = vmin32<UL>(s[i], s[j]), hi = vmax32<UL>(s[i], s[j]); s[i] = lo; s[j] = hi; }()), ...);
}

// sig14 on UL lanes (spec section 3): the two table entries of each lane come from one 32-bit gather.
template <int UL> HGLN_INLINE IV<int32_t, UL> sig14v(IV<int32_t, UL> v) {
  typedef IV<int32_t, UL> V32;
  const V32 vc = vclamp<int32_t, UL>(v, -16384, 16383) + 16384, i = vc >> 5, r = vc & 31;
  alignas(64) int32_t iv[UL], pr[UL];
  st<int32_t, UL>(iv, i);
#if defined(__AVX512F__)
  if (UL == 16) _mm512_store_si512((void*)pr, _mm512_i32gather_epi32(_mm512_load_si512((const void*)iv), (const void*)qs::kSigLut, 2));
  else
#endif
#if defined(__AVX2__)
  if (UL % 8 == 0) for (int h = 0; h < UL; h += 8) _mm256_store_si256((__m256i*)(pr + h), _mm256_i32gather_epi32((const int*)(const void*)qs::kSigLut, _mm256_load_si256((const __m256i*)(iv + h)), 2));
  else
#endif
  for (int k = 0; k < UL; k++) pr[k] = (int32_t)(uint16_t)qs::kSigLut[iv[k]] | ((int32_t)qs::kSigLut[iv[k] + 1] << 16);
  const V32 p = ld<int32_t, UL>(pr), l0 = (p << 16) >> 16, l1 = p >> 16;     // entries are positive int16
  return l0 + (((l1 - l0) * r + 16) >> 5);
}

// Gates and heatmap of one layer (spec sections 3-4). lo: the layer whose bits a latched layer holds (or null).
template <class P, int UL> HGLN_NOINLINE void gates(const LayerQ& L, const QCfg& C, const int32_t* HGLN_RESTRICT s, const LayerQWork* lo, LayerQWork& w) {
  typedef IV<int32_t, UL> V32; typedef IV<uint32_t, UL> VU; typedef IV<int64_t, UL> V64;
  const qs::Fmt& F = C.F; const int D = L.D, sh14 = 16 - F.AB;
  VU z0 = splat<uint32_t, UL>(0), z1 = z0, z2 = z0, z3 = z0;                    // int32 sums, wrap-around
#if defined(__AVX512BW__)
  // ps and H are within int16 (|ps| <= MA, |H| <= MW), so the sums are pairs of int16 products: inputs i and i+1 are
  // interleaved (unpacklo/hi_epi16 within 128-bit lanes) and accumulated with vpdpwssd / vpmaddwd, which wrap mod 2^32
  // like the spec's int32 sums. One 128-bit lane of an accumulator holds 4 units of one bit.
  {
    constexpr int NZ = UL / 8;                                      // zmm per H row (4 x UL int16)
    __m512i aL[NZ], aH[NZ];
    for (int q = 0; q < NZ; q++) aL[q] = aH[q] = _mm512_setzero_si512();
    const __m512i rb = _mm512_set1_epi32((1 << (F.FP - 1)) - 1), one = _mm512_set1_epi32(1), lo_ = _mm512_set1_epi32(-F.MA), hi_ = _mm512_set1_epi32(F.MA);
    const __m128i fp = _mm_cvtsi32_si128(F.FP);
    auto psv = [&](__m512i pv, __m512i sv) {                       // clamp(rne(p s, FP), MA) on 16 int32 lanes
      const __m512i v = _mm512_mullo_epi32(pv, sv);
      const __m512i r = _mm512_sra_epi32(_mm512_add_epi32(_mm512_add_epi32(v, rb), _mm512_and_si512(_mm512_sra_epi32(v, fp), one)), fp);
      return _mm512_min_epi32(_mm512_max_epi32(r, lo_), hi_);
    };
    for (int i = 0; i < D; i += 2) {
      const bool two = i + 1 < D;
      __m512i PL, PH;
      if constexpr (UL == 8) {                                      // rows i and i+1 of p are 16 consecutive int16
        const __m256i pr = two ? _mm256_loadu_si256((const __m256i*)&L.p[(size_t)i * 8]) : _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i*)&L.p[(size_t)i * 8]));
        const __m512i sv = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(s[i])), _mm256_set1_epi32(two ? s[i + 1] : 0), 1);
        const __m256i p16 = _mm512_cvtepi32_epi16(psv(_mm512_cvtepi16_epi32(pr), sv));
        const __m128i q0 = _mm256_castsi256_si128(p16), q1 = _mm256_extracti128_si256(p16, 1);
        PL = _mm512_broadcast_i32x4(_mm_unpacklo_epi16(q0, q1)); PH = _mm512_broadcast_i32x4(_mm_unpackhi_epi16(q0, q1));
      } else {
        const __m256i y0 = _mm512_cvtepi32_epi16(psv(_mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)&L.p[(size_t)i * 16])), _mm512_set1_epi32(s[i])));
        const __m256i y1 = two ? _mm512_cvtepi32_epi16(psv(_mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)&L.p[(size_t)(i + 1) * 16])), _mm512_set1_epi32(s[i + 1]))) : _mm256_setzero_si256();
        PL = _mm512_broadcast_i64x4(_mm256_unpacklo_epi16(y0, y1)); PH = _mm512_broadcast_i64x4(_mm256_unpackhi_epi16(y0, y1));
      }
      const int16_t* HGLN_RESTRICT Ha = &L.H[(size_t)i * 4 * UL];
      for (int q = 0; q < NZ; q++) {
        const __m512i A = _mm512_loadu_si512((const void*)(Ha + 32 * q)), B = two ? _mm512_loadu_si512((const void*)(Ha + 4 * UL + 32 * q)) : _mm512_setzero_si512();
#if defined(__AVX512VNNI__)
        aL[q] = _mm512_dpwssd_epi32(aL[q], _mm512_unpacklo_epi16(A, B), PL); aH[q] = _mm512_dpwssd_epi32(aH[q], _mm512_unpackhi_epi16(A, B), PH);
#else
        aL[q] = _mm512_add_epi32(aL[q], _mm512_madd_epi16(_mm512_unpacklo_epi16(A, B), PL)); aH[q] = _mm512_add_epi32(aH[q], _mm512_madd_epi16(_mm512_unpackhi_epi16(A, B), PH));
#endif
      }
    }
    // 128-bit lane j of accumulator q: Q = 4 q + j -> bit Q / NZ, units 8 (Q % NZ) + 0..3 (aL) and + 4..7 (aH)
    alignas(64) int32_t tl[16 * NZ], th[16 * NZ], zr[4 * UL];
    for (int q = 0; q < NZ; q++) { _mm512_store_si512((void*)(tl + 16 * q), aL[q]); _mm512_store_si512((void*)(th + 16 * q), aH[q]); }
    for (int Q = 0; Q < 4 * NZ; Q++) {
      const int b = Q / NZ, u0 = 8 * (Q % NZ);
      for (int k = 0; k < 4; k++) { zr[b * UL + u0 + k] = tl[4 * Q + k]; zr[b * UL + u0 + 4 + k] = th[4 * Q + k]; }
    }
    z0 = ld<uint32_t, UL>((const uint32_t*)zr); z1 = ld<uint32_t, UL>((const uint32_t*)zr + UL);
    z2 = ld<uint32_t, UL>((const uint32_t*)zr + 2 * UL); z3 = ld<uint32_t, UL>((const uint32_t*)zr + 3 * UL);
  }
#else
  for (int i = 0; i < D; i++) {
    const int16_t* HGLN_RESTRICT Hi = &L.H[(size_t)i * 4 * UL];
    const V32 ps = vclamp<int32_t, UL>(vrne<int32_t, UL>(cvt<V32>(ld<int16_t, UL>(&L.p[(size_t)i * UL])) * s[i], F.FP), -F.MA, F.MA);
    const VU pu = cvt<VU>(ps);
    z0 += cvt<VU>(cvt<V32>(ld<int16_t, UL>(Hi))) * pu; z1 += cvt<VU>(cvt<V32>(ld<int16_t, UL>(Hi + UL))) * pu;
    z2 += cvt<VU>(cvt<V32>(ld<int16_t, UL>(Hi + 2 * UL))) * pu; z3 += cvt<VU>(cvt<V32>(ld<int16_t, UL>(Hi + 3 * UL))) * pu;
  }
#endif
  alignas(64) int32_t z14[4 * UL];
  {
    const VU zs[4] = {z0, z1, z2, z3};
    for (int b = 0; b < 4; b++) {
      const V64 zz = cvt<V64>(cvt<V32>(zs[b]));
      st<int32_t, UL>(z14 + b * UL, cvt<V32>(vclamp<int64_t, UL>(vrne<int64_t, UL>(zz, F.FH), -F.MA, F.MA)) << sh14);
    }
  }
  const bool latched = C.latch && lo;
  int32_t* HGLN_RESTRICT aw = w.a.data();
  {
    const V32 c16 = splat<int32_t, UL>(16384);
    for (int b = 0; b < 4; b++) {                          // all products below are < 2^29: int32 is exact
      const V32 zq = ld<int32_t, UL>(z14 + b * UL);
      V32 av, kap, hold;
      if (latched) {
        const V32 st_ = sig14v<UL>(zq - C.th14), rt = sig14v<UL>(-zq - C.th14), q14 = ld<int32_t, UL>(&lo->a[b * UL]) << sh14;
        const V32 a14 = vclamp<int32_t, UL>(st_ + vrne<int32_t, UL>((c16 - st_ - rt) * q14, 14), 2, 16382);
        av = sh14 ? vrne<int32_t, UL>(a14, sh14) : a14;
        kap = vrne<int32_t, UL>(vrne<int32_t, UL>(st_ * (c16 - st_), 14) * (c16 - q14), 14) + vrne<int32_t, UL>(vrne<int32_t, UL>(rt * (c16 - rt), 14) * q14, 14);
        hold = c16 - st_ - rt;
        av = vclamp<int32_t, UL>(av, 1, F.ONE - 1);
      } else {
        const V32 a14 = sig14v<UL>(zq);
        av = vclamp<int32_t, UL>(sh14 ? vrne<int32_t, UL>(a14, sh14) : a14, 1, F.ONE - 1);
        const V32 aq = av << sh14;
        kap = vrne<int32_t, UL>(aq * (c16 - aq), 14); hold = splat<int32_t, UL>(0);
      }
      st<int32_t, UL>(aw + b * UL, av); st<int32_t, UL>(&w.kap[b * UL], kap); st<int32_t, UL>(&w.hold[b * UL], hold);
    }
  }
  // heatmap: tensor product, bit 0 first, rounded after each product (values <= ONE: no overflow in int32)
  int32_t* HGLN_RESTRICT hs = w.hs.data();
  const V32 one = splat<int32_t, UL>(F.ONE);
  const V32 a0 = ld<int32_t, UL>(aw), a1 = ld<int32_t, UL>(aw + UL), a2 = ld<int32_t, UL>(aw + 2 * UL), a3 = ld<int32_t, UL>(aw + 3 * UL);
  const V32 b0 = one - a0, b1 = one - a1, b2 = one - a2, b3 = one - a3;
  V32 hv[16];
  {
    const V32 q1[4] = {vrne<int32_t, UL>(b0 * b1, F.FA), vrne<int32_t, UL>(a0 * b1, F.FA), vrne<int32_t, UL>(b0 * a1, F.FA), vrne<int32_t, UL>(a0 * a1, F.FA)};
    V32 q2[8];
    for (int c = 0; c < 4; c++) { q2[c] = vrne<int32_t, UL>(q1[c] * b2, F.FA); q2[c + 4] = vrne<int32_t, UL>(q1[c] * a2, F.FA); }
    for (int c = 0; c < 8; c++) { hv[c] = vrne<int32_t, UL>(q2[c] * b3, F.FA); hv[c + 8] = vrne<int32_t, UL>(q2[c] * a3, F.FA); }
    for (int c = 0; c < 16; c++) st<int32_t, UL>(hs + c * UL, hv[c]);
  }
  // top-K corners: rank by key = 16 hs + (15 - c) (all distinct), selected iff rank < K; one division per unit
  int32_t* HGLN_RESTRICT h = w.h.data(); int32_t* HGLN_RESTRICT sel = w.sel.data();
  if (C.K >= 16) {
    for (int q = 0; q < 16 * UL; q++) { h[q] = hs[q]; sel[q] = 1; }
    return;
  }
  // The keys are distinct, so "rank < K" is "key >= the K-th largest key": sort a copy of the keys with Batcher's
  // odd-even merge network (63 compare-exchanges per lane) and take that threshold.
  V32 srt[16];
  for (int c = 0; c < 16; c++) srt[c] = hv[c] * 16 + (15 - c);
  batcher16<UL>(srt, std::make_index_sequence<63>());
  alignas(64) int32_t thv[UL];
  { alignas(64) int32_t sv[16 * UL]; for (int c = 0; c < 16; c++) st<int32_t, UL>(sv + c * UL, srt[c]);   // ascending: K-th largest at 16 - K
    for (int k = 0; k < UL; k++) thv[k] = sv[(16 - C.K) * UL + k] - 1; }
  const V32 thr = ld<int32_t, UL>(thv);
  V32 sm = splat<int32_t, UL>(0), sl[16];
  for (int c = 0; c < 16; c++) { sl[c] = hv[c] * 16 + (15 - c) > thr; sm += hv[c] & sl[c]; st<int32_t, UL>(sel + c * UL, -sl[c]); }
  // R = floor(2^30 / max(sm, 1)): double quotient (exact: sm < 2^20, so the quotient is never within 2^-20 of an
  // integer it is not equal to, far above the double's rounding error), checked by one integer correction
  alignas(64) int32_t smv[UL]; alignas(64) int64_t R[UL];
  st<int32_t, UL>(smv, vclamp<int32_t, UL>(sm, 1, 1 << 30));
  for (int k = 0; k < UL; k++) {
    int64_t r = (int64_t)(1073741824.0 / (double)smv[k]);
    if (r * smv[k] > (1 << 30)) r--; else if ((r + 1) * smv[k] <= (1 << 30)) r++;
    R[k] = r;
  }
  const V64 Rv = ld<int64_t, UL>(R); const int shr = 30 - F.FA;
  for (int c = 0; c < 16; c++)
    st<int32_t, UL>(h + c * UL, cvt<V32>(vrne<int64_t, UL>(cvt<V64>(hv[c] & sl[c]) * Rv, shr)));
}

// Rule outputs, the unit mixture and the unit outputs (spec section 5). in: the layer's n inputs.
template <class P, int UL> HGLN_NOINLINE void rules(const LayerQ& L, const QCfg& C, const int32_t* HGLN_RESTRICT in, LayerQWork& w) {
  typedef IV<int32_t, UL> V32; typedef IV<int32_t, 16> V16; typedef IV<int64_t, 16> V64;
  const qs::Fmt& F = C.F;
  int32_t* HGLN_RESTRICT x = w.x.data();
  for (int j = 0; j < L.n; j++) x[j] = in[j];
  x[L.n] = F.ONE;
  int32_t* HGLN_RESTRICT f = w.f.data(); uint32_t* HGLN_RESTRICT xp = w.xp.data();
  for (int k = 0; k < L.U; k++) {
    const int nc = L.ncol[k], np = (nc + 1) / 2; const int32_t* HGLN_RESTRICT ck = &L.cols[(size_t)k * L.cs];
    int64_t x2 = 0;
    for (int q = 0; q < np; q++) {
      const int32_t v0 = x[ck[2 * q]], v1 = 2 * q + 1 < nc ? x[ck[2 * q + 1]] : 0;
      x2 += (int64_t)v0 * v0 + (int64_t)v1 * v1;
      xp[q] = (uint32_t)(uint16_t)v0 | ((uint32_t)(uint16_t)v1 << 16);
    }
    w.x2[k] = x2;
    alignas(64) int32_t acc[16], fk[16];
    dot16<P>(&L.W[(size_t)k * L.cp * 32], xp, np, acc);
    st<int32_t, 16>(fk, cvt<V16>(vclamp<int64_t, 16>(vrne<int64_t, 16>(cvt<V64>(ld<int32_t, 16>(acc)), F.WB - 2), -F.MA, F.MA)));
    for (int c = 0; c < 16; c++) f[c * UL + k] = fk[c];
  }
  for (int k = L.U; k < UL; k++) { w.x2[k] = 0; for (int c = 0; c < 16; c++) f[c * UL + k] = 0; }
  const int32_t* HGLN_RESTRICT h = w.h.data();
  V32 mix = splat<int32_t, UL>(0);
  for (int c = 0; c < 16; c++) mix += ld<int32_t, UL>(h + c * UL) * ld<int32_t, UL>(f + c * UL);
  alignas(64) int32_t o[UL];
  st<int32_t, UL>(o, vclamp<int32_t, UL>(vrne<int32_t, UL>(mix, F.FF), -F.MA, F.MA));
  for (int k = 0; k < UL; k++) w.o[k] = k < L.U ? o[k] : 0;
}

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#define HGLN_Q_AVX512 1
HGLN_INLINE __m512i rne64_512(__m512i v, int k) {
  const __m128i kk = _mm_cvtsi32_si128(k);
  const __m512i odd = _mm512_and_si512(_mm512_sra_epi64(v, kk), _mm512_set1_epi64(1));
  return _mm512_sra_epi64(_mm512_add_epi64(_mm512_add_epi64(v, _mm512_set1_epi64(((int64_t)1 << (k - 1)) - 1)), odd), kk);
}
HGLN_INLINE __m512i clamp64_512(__m512i v, int64_t lo, int64_t hi) { return _mm512_min_epi64(_mm512_max_epi64(v, _mm512_set1_epi64(lo)), _mm512_set1_epi64(hi)); }
#endif

// qspec::divd(nu, de) on UL lanes, de > 0: the double quotient rounded to nearest even, clamped to +-(2^31 - 1)
template <int UL> HGLN_INLINE IV<int64_t, UL> vdiv_sat32(IV<int64_t, UL> nu, IV<int64_t, UL> de) {
  alignas(64) int64_t n[UL], d[UL], o[UL];
  st<int64_t, UL>(n, nu); st<int64_t, UL>(d, de);
#if defined(HGLN_Q_AVX512)
  for (int h = 0; h < UL; h += 8) {
    const __m512d q = _mm512_roundscale_pd(_mm512_div_pd(_mm512_cvtepi64_pd(_mm512_load_si512((const void*)(n + h))), _mm512_cvtepi64_pd(_mm512_load_si512((const void*)(d + h)))),
                                           _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512d c = _mm512_min_pd(_mm512_max_pd(q, _mm512_set1_pd(-2147483647.0)), _mm512_set1_pd(2147483647.0));
    _mm512_store_si512((void*)(o + h), _mm512_cvttpd_epi64(c));
  }
#else
  for (int k = 0; k < UL; k++) o[k] = qs::divd(n[k], d[k]);
#endif
  return ld<int64_t, UL>(o);
}
template <int UL> HGLN_INLINE IV<int64_t, UL> vclamp32(IV<int64_t, UL> v) { return vclamp<int64_t, UL>(v, -2147483647LL, 2147483647LL); }

// One learning step of one layer (spec section 6). lo: the lower layer's work (latch: its kap feeds the credit).
template <class P, int UL> HGLN_NOINLINE void update(LayerQ& L, const QCfg& C, const QStep& S, const int32_t* HGLN_RESTRICT s, const LayerQWork& w, const LayerQWork* lo) {
  typedef IV<int64_t, UL> V64;
  const qs::Fmt& F = C.F; const int D = L.D, U = L.U;
  const int32_t* HGLN_RESTRICT a = w.a.data(); const int32_t* HGLN_RESTRICT hs = w.hs.data(); const int32_t* HGLN_RESTRICT f = w.f.data();
  const int32_t* HGLN_RESTRICT sel = w.sel.data();
  // gate steps cb [b][lane] (Q30)
  alignas(64) int64_t cb[4][UL];
  {
    V64 tot = splat<int64_t, UL>(0), F1[4] = {tot, tot, tot, tot};
    for (int c = 0; c < 16; c++) {
      const V64 hf = cvt<V64>(ld<int32_t, UL>(hs + c * UL) & -ld<int32_t, UL>(sel + c * UL)) * cvt<V64>(ld<int32_t, UL>(f + c * UL));
      tot += hf;
      if (c & 1) F1[0] += hf;
      if (c & 2) F1[1] += hf;
      if (c & 4) F1[2] += hf;
      if (c & 8) F1[3] += hf;
    }
    alignas(64) int64_t num[4][UL];
    const V64 one = splat<int64_t, UL>(F.ONE);
    for (int b = 0; b < 4; b++) { const V64 av = cvt<V64>(ld<int32_t, UL>(a + b * UL)); st<int64_t, UL>(num[b], F1[b] * (one - av) - (tot - F1[b]) * av); }
    const bool latched = C.latch && lo;
    alignas(64) int64_t gev[UL], acin[4][UL];
    for (int k = 0; k < UL; k++) gev[k] = k < U ? S.ge[k] : 0;
    for (int b = 0; b < 4; b++) for (int k = 0; k < UL; k++) acin[b][k] = S.acc_in && k < U ? S.acc_in[b * UL + k] : 0;
    const V64 ge = ld<int64_t, UL>(gev);
    for (int b = 0; b < 4; b++) {
      const V64 nm = ld<int64_t, UL>(num[b]); V64 Gt;
      if (latched) {                                            // dz(kap) = sat32(div(12 kap num, a (ONE - a) 2^(FF-10)))
        const V64 av = cvt<V64>(ld<int32_t, UL>(a + b * UL)); const int sh = F.FF - 10;
        V64 de = av * (one - av); if (sh > 0) de = de << sh;
        const int nsh = sh < 0 ? -sh : 0;
        Gt = vdiv_sat32<UL>((cvt<V64>(ld<int32_t, UL>(&w.kap[b * UL])) * 12 * nm) << nsh, de);
        if (C.credit && S.acc_out) {
          const V64 dl = vdiv_sat32<UL>((cvt<V64>(ld<int32_t, UL>(&lo->kap[b * UL])) * 12 * nm) << nsh, de);
          const V64 G2 = vclamp32<UL>(vrne<int64_t, UL>(vclamp32<UL>(vrne<int64_t, UL>(dl * cvt<V64>(ld<int32_t, UL>(&w.hold[b * UL])), 14)) * C.cscale, 16));
          alignas(64) int64_t cr[UL];
          st<int64_t, UL>(cr, vclamp<int64_t, UL>(vrne<int64_t, UL>(ge * G2, 18), -kCB, kCB));
          for (int k = 0; k < U; k++) S.acc_out[b * UL + k] += (int32_t)cr[k];
        }
      } else Gt = vclamp32<UL>(vrne<int64_t, UL>(nm * 12, 3 * F.AB - 32));
      st<int64_t, UL>(cb[b], vclamp<int64_t, UL>(vrne<int64_t, UL>(ge * Gt, 18), -kCB, kCB) + ld<int64_t, UL>(acin[b]));
    }
    for (int b = 0; b < 4; b++) for (int k = U; k < UL; k++) cb[b][k] = 0;
  }
  // p step (old H) and hyperplane step (old p), then the renormalisation: the spec's mixed learning steps, float32 on
  // the int16 state, one unit per lane (every lane does the reference's scalar float32 operations, in the same order)
  {
    typedef IV<float, UL> VF; typedef IV<int32_t, UL> VI; typedef IV<int16_t, UL> VS;
    float* HGLN_RESTRICT Hr = reinterpret_cast<float*>(S.scratch);   // D x 4 x UL
    const float KP = std::ldexp(1.0f, -(29 + F.AB)), KH = std::ldexp(1.0f, -(27 + F.AB));
    alignas(64) float cpa[4][UL], cha[4][UL];
    for (int b = 0; b < 4; b++) for (int k = 0; k < UL; k++) { cpa[b][k] = (float)cb[b][k] * KP; cha[b][k] = (float)cb[b][k] * KH; }
    const VF cP0 = ld<float, UL>(cpa[0]), cP1 = ld<float, UL>(cpa[1]), cP2 = ld<float, UL>(cpa[2]), cP3 = ld<float, UL>(cpa[3]);
    const VF cH0 = ld<float, UL>(cha[0]), cH1 = ld<float, UL>(cha[1]), cH2 = ld<float, UL>(cha[2]), cH3 = ld<float, UL>(cha[3]);
    const VF M = splat<float, UL>(12582912.0f);                       // rnf(v) = (v + M) - M
    VF n0 = splat<float, UL>(0.0f), n1 = n0, n2 = n0, n3 = n0;
    for (int i = 0; i < D; i++) {
      int16_t* HGLN_RESTRICT pi = &L.p[(size_t)i * UL]; const int16_t* HGLN_RESTRICT Hi = &L.H[(size_t)i * 4 * UL];
      float* HGLN_RESTRICT Hri = Hr + (size_t)i * 4 * UL;
      const VF sf = splat<float, UL>((float)s[i]), pf = cvt<VF>(cvt<VI>(ld<int16_t, UL>(pi)));
      const VF h0 = cvt<VF>(cvt<VI>(ld<int16_t, UL>(Hi))), h1 = cvt<VF>(cvt<VI>(ld<int16_t, UL>(Hi + UL)));
      const VF h2 = cvt<VF>(cvt<VI>(ld<int16_t, UL>(Hi + 2 * UL))), h3 = cvt<VF>(cvt<VI>(ld<int16_t, UL>(Hi + 3 * UL)));
      const VF ssum = ((cP0 * h0 + cP1 * h1) + cP2 * h2) + cP3 * h3;
      const VF pn = ((pf + sf * ssum) + M) - M;
      st<int16_t, UL>(pi, cvt<VS>(vclamp<int32_t, UL>(cvt<VI>(pn), 0, 3 << F.FP)));
      const VF ps = pf * sf;
      const VF r0 = h0 + cH0 * ps, r1 = h1 + cH1 * ps, r2 = h2 + cH2 * ps, r3 = h3 + cH3 * ps;
      st<float, UL>(Hri, r0); st<float, UL>(Hri + UL, r1); st<float, UL>(Hri + 2 * UL, r2); st<float, UL>(Hri + 3 * UL, r3);
      n0 += r0 * r0; n1 += r1 * r1; n2 += r2 * r2; n3 += r3 * r3;
    }
    // g = 2^FH / sqrt(N2) per row (correctly rounded sqrt and division); N2 = 0: the row keeps its H
    alignas(64) float nn[4][UL], gg[4][UL]; alignas(64) int32_t kp[4][UL];
    st<float, UL>(nn[0], n0); st<float, UL>(nn[1], n1); st<float, UL>(nn[2], n2); st<float, UL>(nn[3], n3);
    for (int b = 0; b < 4; b++) for (int k = 0; k < UL; k++) {
      const bool ok = nn[b][k] > 0.0f;
      gg[b][k] = ok ? (float)(1 << F.FH) / std::sqrt(nn[b][k]) : 0.0f; kp[b][k] = ok ? 0 : -1;
    }
    const VF g[4] = {ld<float, UL>(gg[0]), ld<float, UL>(gg[1]), ld<float, UL>(gg[2]), ld<float, UL>(gg[3])};
    const VI keep[4] = {ld<int32_t, UL>(kp[0]), ld<int32_t, UL>(kp[1]), ld<int32_t, UL>(kp[2]), ld<int32_t, UL>(kp[3])};
    int anykeep = 0;
    for (int b = 0; b < 4; b++) for (int k = 0; k < UL; k++) anykeep |= kp[b][k];
    for (int i = 0; i < D; i++) {
      int16_t* HGLN_RESTRICT Hi = &L.H[(size_t)i * 4 * UL]; const float* HGLN_RESTRICT Hri = Hr + (size_t)i * 4 * UL;
      for (int b = 0; b < 4; b++) {
        VI nh = vclamp<int32_t, UL>(cvt<VI>(((ld<float, UL>(Hri + b * UL) * g[b]) + M) - M), -F.MW, F.MW);
        if (anykeep) nh = vsel<int32_t, UL>(keep[b], cvt<VI>(ld<int16_t, UL>(Hi + b * UL)), nh);   // rows with N2 = 0 keep H
        st<int16_t, UL>(Hi + b * UL, cvt<VS>(nh));
      }
    }
  }
  // rule step (normalised LMS), per unit over its active columns: W' = sat(W + rnf(sW x), MW) (mixed, float32)
  const int Fw = 26 + F.AB - F.WB;
  const int32_t* HGLN_RESTRICT x = w.x.data(); const int32_t* HGLN_RESTRICT h = w.h.data();
  // the per-unit scalars of all units at once (lane = unit): h2, den, c2 = sat32(div(M es << (38 - FA), den)), and
  // sc_c = sat32(rne(c2 h_c, FA)); padding lanes have h = 0 and es = 0, so c2 = sc = 0 there
  alignas(64) int32_t scl[16 * UL]; alignas(64) int64_t esv[UL]; alignas(64) int32_t anyv[UL];
  {
    V64 h2 = splat<int64_t, UL>(0); V64 hc[16];
    for (int c = 0; c < 16; c++) { hc[c] = cvt<V64>(ld<int32_t, UL>(h + c * UL)); h2 += hc[c] * hc[c]; }
    for (int k = 0; k < UL; k++) esv[k] = k < U ? (S.Mq * S.es[k]) * ((int64_t)1 << (38 - F.FA)) : 0;
    const V64 den = vrne<int64_t, UL>(h2, 2 * F.FA - 18) * vrne<int64_t, UL>(ld<int64_t, UL>(w.x2.data()), 2 * F.FA - 12) + (int64_t)1073742;
    const V64 c2 = vdiv_sat32<UL>(ld<int64_t, UL>(esv), den);
    IV<int32_t, UL> any = splat<int32_t, UL>(0);
    for (int c = 0; c < 16; c++) {
      const IV<int32_t, UL> v = cvt<IV<int32_t, UL>>(vclamp32<UL>(vrne<int64_t, UL>(c2 * hc[c], F.FA)));
      st<int32_t, UL>(scl + c * UL, v); any |= v;
    }
    st<int32_t, UL>(anyv, any);
  }
  for (int k = 0; k < U; k++) {
    if (!anyv[k]) continue;
    alignas(64) int32_t sc2[32];
    for (int c = 0; c < 16; c++) sc2[2 * c] = sc2[2 * c + 1] = scl[c * UL + k];
    const int nc = L.ncol[k], np = (nc + 1) / 2; const int32_t* HGLN_RESTRICT ck = &L.cols[(size_t)k * L.cs];
    int16_t* HGLN_RESTRICT Wk = &L.W[(size_t)k * L.cp * 32];
    // 32 elements per slot pair [corner][parity], as two 16-lane halves; x alternates between the pair's two inputs
    // (E x0 + O x1 is exactly x0 or x1 in each lane)
    typedef IV<float, 16> F16; typedef IV<int32_t, 16> I16;
    const float KW = std::ldexp(1.0f, -Fw);
    alignas(64) float swa[32];
    for (int e = 0; e < 32; e++) swa[e] = (float)sc2[e] * KW;
    const F16 sA = ld<float, 16>(swa), sB = ld<float, 16>(swa + 16), M16 = splat<float, 16>(12582912.0f);
    const F16 E = F16{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, O = F16{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    for (int q2 = 0; q2 < np; q2++) {
      const float x0 = (float)x[ck[2 * q2]], x1 = 2 * q2 + 1 < nc ? (float)x[ck[2 * q2 + 1]] : 0.0f;
      const F16 xv = E * x0 + O * x1;
      int16_t* HGLN_RESTRICT Wq = Wk + q2 * 32;
      st<int16_t, 16>(Wq, cvt<IV<int16_t, 16>>(vclamp<int32_t, 16>(cvt<I16>(ld<int16_t, 16>(Wq)) + cvt<I16>(((sA * xv) + M16) - M16), -F.MW, F.MW)));
      st<int16_t, 16>(Wq + 16, cvt<IV<int16_t, 16>>(vclamp<int32_t, 16>(cvt<I16>(ld<int16_t, 16>(Wq + 16)) + cvt<I16>(((sB * xv) + M16) - M16), -F.MW, F.MW)));
    }
  }
  if (!L.pruned) for (int k = 0; k < U; k++) for (int j = 0; j < L.N1; j++) {
    int32_t& Av = L.A[(size_t)k * L.N1 + j];
    Av += (int32_t)qs::rne(1311 * (((int64_t)(x[j] < 0 ? -x[j] : x[j]) << 8) - Av), 16);
  }
}
struct HeaderPolicy {};
}  // namespace qk
}  // namespace detail
}  // namespace hgln

#if defined(HGLN_WITH_FORGE) && !defined(HGLN_SIMD_KERNEL_TU)
#include "hgln_simd.hpp"
#endif

namespace hgln {
// The integer network: same structure and API as Net<T>; R is the caller's real type (float or double) for inputs,
// targets and predictions. Default Params: Params::v5().
template <class R = float> struct NetQ {
  typedef R real_type;
  typedef WorkQ work_type;
  static Params default_params(double g0 = 0.2, double g1 = 0.05) { return Params::v5(g0, g1); }
  int d = 0; LayerQ L[3]; int64_t t = 0; Params P; QCfg C; uint64_t seed = 1; bool pruned_ = false;
  WorkQ ws;
  void init(int d_, uint64_t seed_ = 1, const Params& p = Params::v5()) {
    if (!valid_units(p.units)) throw std::invalid_argument("hgln: Params::units must be 8, 12 or 16");
    if (!qspec::valid_bits(p.wbits) || !qspec::valid_bits(p.abits)) throw std::invalid_argument("hgln: Params::wbits/abits must be 12..16");
    if (p.corners < 1 || p.corners > 16) throw std::invalid_argument("hgln: Params::corners must be 1..16");
    if (p.competence) throw std::invalid_argument("hgln: the integer net has the plain-average mixture only (Params::competence = false)");
    if (!(p.glob >= 0 && p.glob <= 1) || !(p.mu > 0 && p.mu <= 1) || !(p.latch >= 0 && p.latch < 2) || !(p.latch_credit >= 0 && p.latch_credit <= 16) || !(p.tau > 0))
      throw std::invalid_argument("hgln: Params out of range for the integer net (0 <= glob <= 1, 0 < mu <= 1, 0 <= latch < 2, tau > 0)");
    d = d_; P = p; t = 0; seed = seed_; pruned_ = false;
    C.F = qspec::make_fmt(p.abits, p.wbits); C.K = p.corners; C.latch = p.latch > 0; C.credit = p.latch_credit > 0 && C.latch; C.cscale = qspec::round_param(p.latch_credit, 16);
    C.th14 = qspec::round_param(p.latch, C.F.FA) * (1 << (16 - C.F.AB));
    const int U = p.units, UL = U == 8 ? 8 : 16;
    for (int l = 0; l < 3; l++) {
      LayerQ& q = L[l];
      q.U = U; q.UL = UL; q.n = l == 0 ? d : U; q.D = d + 1; q.N1 = q.n + 1; q.nt = q.n <= 4 ? q.n : (6 * q.n + 5) / 10;
      q.cs = q.N1; q.cp = (q.N1 + 1) / 2; q.pruned = false;
      q.H.assign((size_t)q.D * 4 * UL, 0); q.p.assign((size_t)q.D * UL, 0); q.W.assign((size_t)U * q.cp * 32, 0);
      q.A.assign((size_t)U * q.N1, (int32_t)qspec::div_rne((int64_t)1 << (C.F.FA + 8), 1000));
      q.cols.assign((size_t)U * q.cs, 0); q.ncol.assign(U, q.N1);
      for (int k = 0; k < U; k++) {
        for (int j = 0; j < q.N1; j++) q.cols[(size_t)k * q.cs + j] = j;
        for (int i = 0; i < q.D; i++) q.p[(size_t)i * UL + k] = (int16_t)(1 << C.F.FP);
        for (int b = 0; b < 4; b++) {
          std::vector<int32_t> v(q.D);
          for (int i = 0; i < q.D; i++) v[i] = qspec::init_component(seed, l, k, b, i, q.D);
          qspec::init_normalise(q.D, C.F, [&](int i) { return v[i]; }, [&](int i, int32_t hv) { q.H[((size_t)i * 4 + b) * UL + k] = (int16_t)hv; });
        }
      }
    }
    ws.init(*this);
  }
  int units() const { return L[0].U; }
  bool pruned() const { return pruned_; }
  int lower(int l) const { return l == 0 ? -1 : (P.fanout ? 0 : l - 1); }
  double output(int i) const { return std::ldexp((double)ws.o[i], -C.F.FA); }   // unit i of the last predict()
  R out() const { return (R)((double)ws.S / (double)((int64_t)3 * units() << C.F.FA)); }

  // ---- kernels (FORGE build: the table chosen for this CPU; otherwise inline)
  void k_gates(int l, const int32_t* s, const LayerQWork* lo, LayerQWork& w) const {
#if defined(HGLN_WITH_FORGE) && !defined(HGLN_SIMD_KERNEL_TU)
    simd::active().q.gates[L[l].UL == 8 ? 0 : 1](L[l], C, s, lo, w);
#else
    if (L[l].UL == 8) detail::qk::gates<detail::qk::HeaderPolicy, 8>(L[l], C, s, lo, w); else detail::qk::gates<detail::qk::HeaderPolicy, 16>(L[l], C, s, lo, w);
#endif
  }
  void k_rules(int l, const int32_t* in, LayerQWork& w) const {
#if defined(HGLN_WITH_FORGE) && !defined(HGLN_SIMD_KERNEL_TU)
    simd::active().q.rules[L[l].UL == 8 ? 0 : 1](L[l], C, in, w);
#else
    if (L[l].UL == 8) detail::qk::rules<detail::qk::HeaderPolicy, 8>(L[l], C, in, w); else detail::qk::rules<detail::qk::HeaderPolicy, 16>(L[l], C, in, w);
#endif
  }
  void k_update(int l, const QStep& S, const int32_t* s, const LayerQWork& w, const LayerQWork* lo) {
#if defined(HGLN_WITH_FORGE) && !defined(HGLN_SIMD_KERNEL_TU)
    simd::active().q.update[L[l].UL == 8 ? 0 : 1](L[l], C, S, s, w, lo);
#else
    if (L[l].UL == 8) detail::qk::update<detail::qk::HeaderPolicy, 8>(L[l], C, S, s, w, lo); else detail::qk::update<detail::qk::HeaderPolicy, 16>(L[l], C, S, s, w, lo);
#endif
  }

  // Forward pass into an arbitrary Work (const: many threads may predict at once, each with its own Work).
  R infer(const R* xin, WorkQ& w) const {
    const qspec::Fmt& F = C.F;
    int32_t* xq = w.lw[0].x.data();                          // layer 1's inputs are the quantised x
    int64_t nn = (int64_t)F.ONE * F.ONE;
    for (int j = 0; j < d; j++) { xq[j] = qspec::quantise((double)xin[j], F.FA, F.MA); nn += (int64_t)xq[j] * xq[j]; }
    const int64_t g = (int64_t)qspec::isqrt((uint64_t)nn << 6), rcp = ((int64_t)1 << (34 + F.FA)) / g;
    int32_t* s = w.s.data();
    for (int j = 0; j < d; j++) s[j] = (int32_t)qspec::rne(xq[j] * rcp, 31);
    s[d] = (int32_t)qspec::rne((int64_t)F.ONE * rcp, 31);
    for (int l = 0; l < 3; l++) { const int lo = lower(l); k_gates(l, s, lo < 0 ? nullptr : &w.lw[lo], w.lw[l]); }   // gates first
    k_rules(0, xq, w.lw[0]);
    k_rules(1, w.lw[0].o.data(), w.lw[1]);
    k_rules(2, w.lw[P.fanout ? 0 : 1].o.data(), w.lw[2]);
    const int U = units(); int32_t S = 0;
    for (int l = 0; l < 3; l++) for (int k = 0; k < U; k++) { w.o[l * U + k] = w.lw[l].o[k]; S += w.lw[l].o[k]; }
    w.S = S;
    return (R)((double)S / (double)((int64_t)3 * U << F.FA));
  }
  R predict(const R* x) { return infer(x, ws); }
  void learn(R y) {                                          // must follow predict() on the same sample
    const qspec::Fmt& F = C.F; const int U = units(), NU = 3 * U, UL = L[0].UL;
    const int64_t yq = qspec::quantise((double)y, F.FA, 1 << (F.FA + 6));
    const int64_t eg = qspec::div_rne(NU * yq - ws.S, NU), G = qspec::round_param(P.glob, 16);
    const int64_t glq = qspec::gl_q24(t, P.glr0, P.glr1, P.tau);
    alignas(64) int32_t es[3 * kMaxUnits], ge[3 * kMaxUnits], acc[3][4 * 16];
    for (int i = 0; i < NU; i++) {
      const int64_t e = yq - ws.o[i], esi = qspec::rne((65536 - G) * e + G * eg, 16);
      es[i] = (int32_t)esi;
      ge[i] = qspec::sat32(qspec::rne(glq * (P.gate_err == 2 ? eg : (P.gate_err == 1 ? e : esi)), F.FA));
    }
    for (auto& a : acc) for (auto& v : a) v = 0;
    scratch_();
    QStep S; S.Mq = qspec::round_param(P.mu, 16); S.scratch = scr_.data();
    for (int l = 2; l >= 0; l--) {                           // upper layers first: latch credit reaches layer 1 now
      const int lo = lower(l);
      S.layer = (uint32_t)l; S.es = es + l * U; S.ge = ge + l * U; S.acc_in = acc[l];
      S.acc_out = (C.credit && lo >= 0) ? acc[lo] : nullptr;
      k_update(l, S, ws.s.data(), ws.lw[l], lo < 0 ? nullptr : &ws.lw[lo]);
    }
    (void)UL;
    t++;
    if (!pruned_ && t >= P.warm && t % 64 == 0) { for (auto& l : L) prune(l); pruned_ = true; }
  }
  // Load layer l's state from the dense reference layout (stored integers; the integer GPU kernel's layout):
  // H[(k 4 + b) D + i], p[k D + i], W[(k 16 + c) N1 + j], M and A [k N1 + j] (M != 0: input j active). pruned: the
  // one-off prune has run (W is then compacted to the active columns, ascending with the bias last, and A is freed).
  // Set t and pruned_ of the network to match (pruned_ = every layer pruned).
  void import_dense(int l, const int32_t* H, const int32_t* p, const int32_t* W, const int32_t* M, const int32_t* A, bool pruned) {
    LayerQ& q = L[l]; const int U = q.U, UL = q.UL, D = q.D, N1 = q.N1;
    for (int k = 0; k < U; k++) for (int i = 0; i < D; i++) {
      q.p[(size_t)i * UL + k] = (int16_t)p[(size_t)k * D + i];
      for (int b = 0; b < 4; b++) q.H[((size_t)i * 4 + b) * UL + k] = (int16_t)H[((size_t)k * 4 + b) * D + i];
    }
    int cs = 0;
    for (int k = 0; k < U; k++) {
      int nc = 0;
      for (int j = 0; j < N1; j++) if (!pruned || M[(size_t)k * N1 + j]) nc++;
      q.ncol[k] = nc; cs = std::max(cs, nc);
    }
    q.cs = cs; q.cp = (cs + 1) / 2;
    q.cols.assign((size_t)U * cs, 0); q.W.assign((size_t)U * q.cp * 32, 0);
    for (int k = 0; k < U; k++) {
      int s = 0;
      for (int j = 0; j < N1; j++) {
        if (pruned && !M[(size_t)k * N1 + j]) continue;
        q.cols[(size_t)k * cs + s] = j;
        for (int c = 0; c < 16; c++) q.W[q.widx(k, s, c)] = (int16_t)W[((size_t)k * 16 + c) * N1 + j];
        s++;
      }
    }
    q.pruned = pruned;
    if (pruned) q.A.reset();
    else { q.A.assign((size_t)U * N1, 0); for (size_t i = 0; i < (size_t)U * N1; i++) q.A[i] = A[i]; }
  }
  R step(const R* x, R y) { R p = predict(x); learn(y); return p; }
  size_t model_bytes() const { return L[0].model_bytes() + L[1].model_bytes() + L[2].model_bytes() + sizeof(*this); }

 private:
  std::vector<int32_t> scr_;
  void scratch_() { const size_t n = (size_t)L[0].D * 4 * L[0].UL; if (scr_.size() < n) scr_.assign(n, 0); }   // Hr
  // the one-off prune (spec section 6): compacts W to the kept columns and frees A
  static void prune(LayerQ& q) {
    const int U = q.U, n = q.n;
    std::vector<std::vector<int>> keep(U);
    int newcs = 0;
    for (int k = 0; k < U; k++) {
      std::vector<int> act; std::vector<int64_t> sc(n, 0); std::vector<char> drop(n, 0);
      for (int s = 0; s < q.ncol[k]; s++) { const int j = q.cols[(size_t)k * q.cs + s]; if (j < n) act.push_back(j); }
      if ((int)act.size() > q.nt) {
        for (int s = 0; s < q.ncol[k]; s++) {
          const int j = q.cols[(size_t)k * q.cs + s]; if (j >= n) continue;
          int64_t sw = 0; for (int c = 0; c < 16; c++) { const int v = q.W[q.widx(k, s, c)]; sw += v < 0 ? -v : v; }
          sc[j] = sw * q.A[(size_t)k * q.N1 + j];
        }
        std::stable_sort(act.begin(), act.end(), [&](int a, int b) { return sc[a] < sc[b]; });
        for (int i = 0; i < (int)act.size() - q.nt; i++) drop[act[i]] = 1;
      }
      for (int s = 0; s < q.ncol[k]; s++) { const int j = q.cols[(size_t)k * q.cs + s]; if (j >= n || !drop[j]) keep[k].push_back(s); }
      newcs = std::max(newcs, (int)keep[k].size());
    }
    const int newcp = (newcs + 1) / 2;
    detail::ABuf<int16_t> W2; W2.assign((size_t)U * newcp * 32, 0);
    detail::ABuf<int32_t> c2; c2.assign((size_t)U * newcs, 0);
    for (int k = 0; k < U; k++) {
      for (int s2 = 0; s2 < (int)keep[k].size(); s2++) {
        const int s = keep[k][s2];
        c2[(size_t)k * newcs + s2] = q.cols[(size_t)k * q.cs + s];
        for (int c = 0; c < 16; c++) W2[((size_t)k * newcp + (s2 >> 1)) * 32 + c * 2 + (s2 & 1)] = q.W[q.widx(k, s, c)];
      }
      q.ncol[k] = (int32_t)keep[k].size();
    }
    q.W = std::move(W2); q.cols = std::move(c2); q.cs = newcs; q.cp = newcp;
    q.A.reset(); q.pruned = true;
  }
};
}  // namespace hgln
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#pragma GCC diagnostic ignored "-Wpsabi"
#endif
