// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_q.hpp - HGLN version 5.0 integer datapath: hgln::NetQ<R>, the same public API as hgln::Net<T> (predict /
// learn / infer with a caller-owned Work, Params, Classifier, run_streams) on int16 state and activations.
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
//     accumulator over column pairs (AVX-512 VNNI vpdpwssd / AVX-512BW and AVX2 vpmaddwd), and the 32 int16 of a slot
//     pair line up with the 32 16-bit halves of the 16 Philox blocks that round them (one block per corner);
//   * after the prune the inactive columns are removed from W and the activity trace A is freed.
#pragma once
#include "hgln.hpp"
#include "hgln_qspec.hpp"
#include <type_traits>
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
  int64_t Mq = 0; uint32_t layer = 0, tl = 0, th = 0, k0 = 0, k1 = 0;
  const int32_t* es = nullptr;                                // U rule errors (Q(AB-2))
  const int32_t* ge = nullptr;                                // U gate errors times the gate rate (Q24)
  const int32_t* acc_in = nullptr;                            // 4 x UL gate steps routed here by upper layers (or null)
  int32_t* acc_out = nullptr;                                 // 4 x UL routed to the lower layer (latch credit), or null
  int32_t* scratch = nullptr;                                 // D x 4 x UL
  uint32_t* c0 = nullptr;                                     // Philox counters, 16 per group (NetQ::kRndGroups groups)
  uint32_t* rnd = nullptr;                                    // Philox output, 64 words per group
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
#define HGLN_QIV_OP(op)                                                                                                    \
  template <class T, int N> inline IV<T, N> operator op(const IV<T, N>& a, const IV<T, N>& b) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = (T)(a.v[i] op b.v[i]); return r; } \
  template <class T, int N> inline IV<T, N> operator op(const IV<T, N>& a, T s) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = (T)(a.v[i] op s); return r; } \
  template <class T, int N> inline IV<T, N> operator op(T s, const IV<T, N>& a) { IV<T, N> r; for (int i = 0; i < N; i++) r.v[i] = (T)(s op a.v[i]); return r; } \
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
  IV<T, N> r = vsel<T, N>(v < L, L, v); return vsel<T, N>(r > H, H, r);
}
template <class T, int N> HGLN_INLINE IV<T, N> vrne(IV<T, N> v, int k) {     // spec rne, k >= 1 (no overflow by the callers' bounds)
  return (v + (T)(((T)1 << (k - 1)) - 1) + ((v >> k) & (T)1)) >> k;
}
template <int N> HGLN_INLINE IV<int64_t, N> vsr(IV<int64_t, N> Y, int F, IV<int64_t, N> r16) {   // spec sr, F >= 16
  return (Y >> F) - (r16 < ((Y >> (F - 16)) & (int64_t)0xFFFF));   // a true compare is -1
}

// ---- Philox4x32-10 on N lanes: counter (base + lane, c1, c2, c3), key (k0, k1); out[w][lane].
#if defined(__AVX512F__)
HGLN_INLINE __m512i philox_mulhilo(__m512i x, __m512i m, __m512i& lo) {
  const __m512i ev = _mm512_mul_epu32(x, m), od = _mm512_mul_epu32(_mm512_srli_epi64(x, 32), m);
  lo = _mm512_permutex2var_epi32(ev, _mm512_setr_epi32(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30), od);
  return _mm512_permutex2var_epi32(ev, _mm512_setr_epi32(1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31), od);
}
#endif
#if defined(__AVX2__)
HGLN_INLINE __m256i philox_mulhilo(__m256i x, __m256i m, __m256i& lo) {
  const __m256i ev = _mm256_mul_epu32(x, m), od = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), m);
  lo = _mm256_blend_epi32(ev, _mm256_slli_epi64(od, 32), 0xAA);
  return _mm256_blend_epi32(_mm256_srli_epi64(ev, 32), od, 0xAA);
}
#endif
template <class P, int N> HGLN_INLINE void philox_lanes(uint32_t base, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1, uint32_t (*out)[N]) {
#if defined(__AVX512F__)
  if (N == 16) {
    const __m512i M0 = _mm512_set1_epi32((int)0xD2511F53u), M1 = _mm512_set1_epi32((int)0xCD9E8D57u);
    __m512i a = _mm512_add_epi32(_mm512_set1_epi32((int)base), _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    __m512i b = _mm512_set1_epi32((int)c1), c = _mm512_set1_epi32((int)c2), d = _mm512_set1_epi32((int)c3);
    for (int r = 0; r < 10; r++) {
      __m512i lo0, lo1; const __m512i hi0 = philox_mulhilo(a, M0, lo0), hi1 = philox_mulhilo(c, M1, lo1);
      a = _mm512_xor_si512(_mm512_xor_si512(hi1, b), _mm512_set1_epi32((int)k0)); c = _mm512_xor_si512(_mm512_xor_si512(hi0, d), _mm512_set1_epi32((int)k1));
      b = lo1; d = lo0; k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    _mm512_storeu_si512((void*)out[0], a); _mm512_storeu_si512((void*)out[1], b); _mm512_storeu_si512((void*)out[2], c); _mm512_storeu_si512((void*)out[3], d);
    return;
  }
#endif
#if defined(__AVX2__)
  if (N % 8 == 0) {
    const __m256i M0 = _mm256_set1_epi32((int)0xD2511F53u), M1 = _mm256_set1_epi32((int)0xCD9E8D57u);
    for (int h = 0; h < N; h += 8) {
      __m256i a = _mm256_add_epi32(_mm256_set1_epi32((int)(base + (uint32_t)h)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
      __m256i b = _mm256_set1_epi32((int)c1), c = _mm256_set1_epi32((int)c2), d = _mm256_set1_epi32((int)c3);
      uint32_t q0 = k0, q1 = k1;
      for (int r = 0; r < 10; r++) {
        __m256i lo0, lo1; const __m256i hi0 = philox_mulhilo(a, M0, lo0), hi1 = philox_mulhilo(c, M1, lo1);
        a = _mm256_xor_si256(_mm256_xor_si256(hi1, b), _mm256_set1_epi32((int)q0)); c = _mm256_xor_si256(_mm256_xor_si256(hi0, d), _mm256_set1_epi32((int)q1));
        b = lo1; d = lo0; q0 += 0x9E3779B9u; q1 += 0xBB67AE85u;
      }
      _mm256_storeu_si256((__m256i*)(out[0] + h), a); _mm256_storeu_si256((__m256i*)(out[1] + h), b);
      _mm256_storeu_si256((__m256i*)(out[2] + h), c); _mm256_storeu_si256((__m256i*)(out[3] + h), d);
    }
    return;
  }
#endif
  uint32_t a[N], b[N], c[N], d[N];
  for (int l = 0; l < N; l++) { a[l] = base + (uint32_t)l; b[l] = c1; c[l] = c2; d[l] = c3; }
  for (int r = 0; r < 10; r++) {
    for (int l = 0; l < N; l++) {
      const uint64_t p0 = (uint64_t)0xD2511F53u * a[l], p1 = (uint64_t)0xCD9E8D57u * c[l];
      const uint32_t na = (uint32_t)(p1 >> 32) ^ b[l] ^ k0, nc = (uint32_t)(p0 >> 32) ^ d[l] ^ k1;
      b[l] = (uint32_t)p1; d[l] = (uint32_t)p0; a[l] = na; c[l] = nc;
    }
    k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
  }
  for (int l = 0; l < N; l++) { out[0][l] = a[l]; out[1][l] = b[l]; out[2][l] = c[l]; out[3][l] = d[l]; }
}
// one 16-bit half of each lane's word: (w >> (16 half)) & 0xFFFF, as int64 lanes
template <int N> HGLN_INLINE IV<int64_t, N> half_lanes(const uint32_t* w, int half) {
  return cvt<IV<int64_t, N>>((ld<uint32_t, N>(w) >> (16 * half)) & 0xFFFFu);
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
  for (int i = 0; i < D; i++) {
    const int16_t* HGLN_RESTRICT Hi = &L.H[(size_t)i * 4 * UL];
    const V32 ps = vclamp<int32_t, UL>(vrne<int32_t, UL>(cvt<V32>(ld<int16_t, UL>(&L.p[(size_t)i * UL])) * s[i], F.FP), -F.MA, F.MA);
    const VU pu = cvt<VU>(ps);
    z0 += cvt<VU>(cvt<V32>(ld<int16_t, UL>(Hi))) * pu; z1 += cvt<VU>(cvt<V32>(ld<int16_t, UL>(Hi + UL))) * pu;
    z2 += cvt<VU>(cvt<V32>(ld<int16_t, UL>(Hi + 2 * UL))) * pu; z3 += cvt<VU>(cvt<V32>(ld<int16_t, UL>(Hi + 3 * UL))) * pu;
  }
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
  V32 key[16], rank[16];
  for (int c = 0; c < 16; c++) { key[c] = hv[c] * 16 + (15 - c); rank[c] = splat<int32_t, UL>(0); }
  for (int c = 0; c < 16; c++)
    for (int c2 = c + 1; c2 < 16; c2++) {
      const V32 gt = key[c2] > key[c];                                             // -1 if c2 ranks above c
      rank[c] -= gt; rank[c2] += gt + 1;
    }
  const V32 Kv = splat<int32_t, UL>(C.K);
  V32 sm = splat<int32_t, UL>(0), sl[16];
  for (int c = 0; c < 16; c++) { sl[c] = rank[c] < Kv; sm += hv[c] & sl[c]; st<int32_t, UL>(sel + c * UL, -sl[c]); }
  alignas(64) int32_t smv[UL]; alignas(64) int64_t R[UL];
  st<int32_t, UL>(smv, sm);
  for (int k = 0; k < UL; k++) R[k] = (int64_t)((1u << 30) / (uint32_t)(smv[k] > 1 ? smv[k] : 1));
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

// ---- many Philox groups at once: group g has 16 lanes with counters c0[16 g + lane] and (c1, c2, c3); out[(4 g + w) * 16
// + lane]. Four groups are computed together (independent chains hide the multiply latency).
#if defined(__AVX512F__)
HGLN_INLINE void philox_x4_512(const uint32_t* c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1, uint32_t* out, int n) {
  const __m512i M0 = _mm512_set1_epi32((int)0xD2511F53u), M1 = _mm512_set1_epi32((int)0xCD9E8D57u);
  __m512i a[4], b[4], c[4], d[4];
  for (int g = 0; g < 4; g++) { a[g] = _mm512_loadu_si512((const void*)(c0 + 16 * (g < n ? g : 0))); b[g] = _mm512_set1_epi32((int)c1); c[g] = _mm512_set1_epi32((int)c2); d[g] = _mm512_set1_epi32((int)c3); }
  for (int r = 0; r < 10; r++) {
    const __m512i K0 = _mm512_set1_epi32((int)k0), K1 = _mm512_set1_epi32((int)k1);
    for (int g = 0; g < 4; g++) {
      __m512i lo0, lo1; const __m512i hi0 = philox_mulhilo(a[g], M0, lo0), hi1 = philox_mulhilo(c[g], M1, lo1);
      a[g] = _mm512_xor_si512(_mm512_xor_si512(hi1, b[g]), K0); c[g] = _mm512_xor_si512(_mm512_xor_si512(hi0, d[g]), K1);
      b[g] = lo1; d[g] = lo0;
    }
    k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
  }
  for (int g = 0; g < n; g++) {
    uint32_t* o = out + 64 * g;
    _mm512_storeu_si512((void*)o, a[g]); _mm512_storeu_si512((void*)(o + 16), b[g]); _mm512_storeu_si512((void*)(o + 32), c[g]); _mm512_storeu_si512((void*)(o + 48), d[g]);
  }
}
#endif
template <class P> HGLN_INLINE void philox_groups(const uint32_t* c0, int ng, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1, uint32_t* out) {
#if defined(__AVX512F__)
  for (int g = 0; g < ng; g += 4) philox_x4_512(c0 + 16 * g, c1, c2, c3, k0, k1, out + 64 * g, ng - g < 4 ? ng - g : 4);
#else
  for (int g = 0; g < ng; g++) {
    uint32_t a[16], b[16], c[16], d[16], q0 = k0, q1 = k1;
    for (int l = 0; l < 16; l++) { a[l] = c0[16 * g + l]; b[l] = c1; c[l] = c2; d[l] = c3; }
    for (int r = 0; r < 10; r++) {
      for (int l = 0; l < 16; l++) {
        const uint64_t p0 = (uint64_t)0xD2511F53u * a[l], p1 = (uint64_t)0xCD9E8D57u * c[l];
        const uint32_t na = (uint32_t)(p1 >> 32) ^ b[l] ^ q0, nc = (uint32_t)(p0 >> 32) ^ d[l] ^ q1;
        b[l] = (uint32_t)p1; d[l] = (uint32_t)p0; a[l] = na; c[l] = nc;
      }
      q0 += 0x9E3779B9u; q1 += 0xBB67AE85u;
    }
    uint32_t* o = out + 64 * g;
    for (int l = 0; l < 16; l++) { o[l] = a[l]; o[16 + l] = b[l]; o[32 + l] = c[l]; o[48 + l] = d[l]; }
  }
#endif
}
// Groups of the unit-lane arrays (p: array 2, H: array 1) of one layer step: UL = 16 lanes = units; UL = 8: two
// sub-blocks of 8 units per group (p: two i-batches; H: two bits).
template <int UL> struct RndMap {
  static int groups_p(int D) { const int nb = (D + 7) / 8; return UL == 16 ? nb : (nb + 1) / 2; }
  static int groups_h(int D) { const int nb = (D + 7) / 8; return UL == 16 ? 4 * nb : 2 * nb; }
  static void fill_p(int D, uint32_t* c0) {
    for (int g = 0; g < groups_p(D); g++) for (int l = 0; l < 16; l++)
      c0[16 * g + l] = UL == 16 ? (uint32_t)(g * 16 + l) : (uint32_t)((2 * g + (l >> 3)) * 16 + (l & 7));
  }
  static void fill_h(int D, uint32_t* c0) {
    for (int g = 0; g < groups_h(D); g++) for (int l = 0; l < 16; l++)
      c0[16 * g + l] = UL == 16 ? (uint32_t)(g * 16 + l) : (uint32_t)(((g >> 1) * 4 + 2 * (g & 1) + (l >> 3)) * 16 + (l & 7));
  }
  // the 4 words of lanes [lane0, lane0 + UL) for input i (and bit b)
  static const uint32_t* word_p(const uint32_t* R, int i) { const int bt = i >> 3; return UL == 16 ? R + 64 * bt + 16 * ((i >> 1) & 3) : R + 64 * (bt >> 1) + 16 * ((i >> 1) & 3) + 8 * (bt & 1); }
  static const uint32_t* word_h(const uint32_t* R, int i, int b) { const int bt = i >> 3; return UL == 16 ? R + 64 * (bt * 4 + b) + 16 * ((i >> 1) & 3) : R + 64 * (bt * 2 + (b >> 1)) + 16 * ((i >> 1) & 3) + 8 * (b & 1); }
};

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#define HGLN_Q_AVX512 1
HGLN_INLINE __m512i rne64_512(__m512i v, int k) {
  const __m128i kk = _mm_cvtsi32_si128(k);
  const __m512i odd = _mm512_and_si512(_mm512_sra_epi64(v, kk), _mm512_set1_epi64(1));
  return _mm512_sra_epi64(_mm512_add_epi64(_mm512_add_epi64(v, _mm512_set1_epi64(((int64_t)1 << (k - 1)) - 1)), odd), kk);
}
HGLN_INLINE __m512i sr64_512(__m512i Y, int F, __m512i r16) {
  const __m512i q = _mm512_sra_epi64(Y, _mm_cvtsi32_si128(F));
  const __m512i f16 = _mm512_and_si512(_mm512_srl_epi64(Y, _mm_cvtsi32_si128(F - 16)), _mm512_set1_epi64(0xFFFF));
  return _mm512_mask_add_epi64(q, _mm512_cmplt_epi64_mask(r16, f16), q, _mm512_set1_epi64(1));
}
HGLN_INLINE __m512i clamp64_512(__m512i v, int64_t lo, int64_t hi) { return _mm512_min_epi64(_mm512_max_epi64(v, _mm512_set1_epi64(lo)), _mm512_set1_epi64(hi)); }
HGLN_INLINE __m512i half8_512(const uint32_t* w, int half) {   // 8 lanes: (w >> 16 half) & 0xFFFF as int64
  return _mm512_and_si512(_mm512_srl_epi64(_mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i*)w)), _mm_cvtsi32_si128(16 * half)), _mm512_set1_epi64(0xFFFF));
}
#endif

// sat32(div_rne(nu, de)) on UL lanes, de > 0 (spec 'div'): exact. AVX-512: double estimate, integer correction.
template <int UL> HGLN_INLINE IV<int64_t, UL> vdiv_sat32(IV<int64_t, UL> nu, IV<int64_t, UL> de) {
  alignas(64) int64_t n[UL], d[UL], o[UL];
  st<int64_t, UL>(n, nu); st<int64_t, UL>(d, de);
#if defined(HGLN_Q_AVX512)
  for (int h = 0; h < UL; h += 8) {
    const __m512i N = _mm512_load_si512((const void*)(n + h)), Dv = _mm512_load_si512((const void*)(d + h)), one = _mm512_set1_epi64(1), z = _mm512_setzero_si512();
    const __m512d qa = _mm512_div_pd(_mm512_cvtepi64_pd(N), _mm512_cvtepi64_pd(Dv)), lim = _mm512_set1_pd(1099511627776.0);   // 2^40
    const __mmask8 hi = _mm512_cmp_pd_mask(qa, lim, _CMP_GE_OQ), lw = _mm512_cmp_pd_mask(qa, _mm512_sub_pd(_mm512_setzero_pd(), lim), _CMP_LE_OQ);
    __m512i q = _mm512_cvttpd_epi64(_mm512_min_pd(_mm512_max_pd(qa, _mm512_sub_pd(_mm512_setzero_pd(), lim)), lim));
    __m512i r = _mm512_sub_epi64(N, _mm512_mullo_epi64(q, Dv));
    for (int it = 0; it < 2; it++) {
      const __mmask8 neg = _mm512_cmplt_epi64_mask(r, z);
      q = _mm512_mask_sub_epi64(q, neg, q, one); r = _mm512_mask_add_epi64(r, neg, r, Dv);
      const __mmask8 big = _mm512_cmpge_epi64_mask(r, Dv);
      q = _mm512_mask_add_epi64(q, big, q, one); r = _mm512_mask_sub_epi64(r, big, r, Dv);
    }
    const __m512i r2 = _mm512_add_epi64(r, r);
    const __mmask8 up = _mm512_cmpgt_epi64_mask(r2, Dv) | (_mm512_cmpeq_epi64_mask(r2, Dv) & _mm512_test_epi64_mask(q, one));
    q = _mm512_mask_add_epi64(q, up, q, one);
    q = _mm512_min_epi64(_mm512_max_epi64(q, _mm512_set1_epi64(-2147483647LL)), _mm512_set1_epi64(2147483647LL));
    q = _mm512_mask_blend_epi64(hi, q, _mm512_set1_epi64(2147483647LL)); q = _mm512_mask_blend_epi64(lw, q, _mm512_set1_epi64(-2147483647LL));
    _mm512_store_si512((void*)(o + h), q);
  }
#else
  for (int k = 0; k < UL; k++) o[k] = qs::sat32(qs::div_rne(n[k], d[k]));
#endif
  return ld<int64_t, UL>(o);
}
template <int UL> HGLN_INLINE IV<int64_t, UL> vclamp32(IV<int64_t, UL> v) { return vclamp<int64_t, UL>(v, -2147483647LL, 2147483647LL); }

// One learning step of one layer (spec section 6). lo: the lower layer's work (latch: its kap feeds the credit).
template <class P, int UL> HGLN_NOINLINE void update(LayerQ& L, const QCfg& C, const QStep& S, const int32_t* HGLN_RESTRICT s, const LayerQWork& w, const LayerQWork* lo) {
  typedef IV<int64_t, UL> V64;
#if !defined(HGLN_Q_AVX512)
  typedef IV<int32_t, UL> V32;
#endif
  const qs::Fmt& F = C.F; const int D = L.D, U = L.U;
  const int32_t* HGLN_RESTRICT a = w.a.data(); const int32_t* HGLN_RESTRICT hs = w.hs.data(); const int32_t* HGLN_RESTRICT f = w.f.data();
  const int32_t* HGLN_RESTRICT sel = w.sel.data();
  // all random words of this step: p and H groups, then W groups per unit
  const int gp = RndMap<UL>::groups_p(D), gh = RndMap<UL>::groups_h(D);
  uint32_t* HGLN_RESTRICT c0 = S.c0; uint32_t* HGLN_RESTRICT R = S.rnd;
  RndMap<UL>::fill_p(D, c0); RndMap<UL>::fill_h(D, c0 + 16 * gp);
  philox_groups<P>(c0, gp, 4 * S.layer + qs::kArrP, S.tl, S.th, S.k0, S.k1, R);
  philox_groups<P>(c0 + 16 * gp, gh, 4 * S.layer + qs::kArrH, S.tl, S.th, S.k0, S.k1, R + 64 * gp);
  const uint32_t* Rp = R; const uint32_t* Rh = R + 64 * gp;
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
  // p step (old H) and hyperplane step (old p); Hr kept in scratch, then exact renormalisation
  {
    int32_t* HGLN_RESTRICT Hr = S.scratch;
    const int Fp = 29 + F.AB, Fh = 19 + F.AB, Fr = 56 - F.WB;
    alignas(64) int64_t rc[4][UL];
#if !defined(HGLN_Q_AVX512)
    alignas(64) int64_t N2[4][UL];
#endif
#if defined(HGLN_Q_AVX512)
    constexpr int NH = UL / 8;                                   // 8-lane int64 halves
    __m512i cbv[4][NH], n2v[4][NH];
    for (int b = 0; b < 4; b++) for (int hh = 0; hh < NH; hh++) { cbv[b][hh] = _mm512_load_si512((const void*)(cb[b] + 8 * hh)); n2v[b][hh] = _mm512_setzero_si512(); }
    for (int i = 0; i < D; i++) {
      const __m512i si = _mm512_set1_epi64(s[i]);
      int16_t* HGLN_RESTRICT pi = &L.p[(size_t)i * UL]; const int16_t* HGLN_RESTRICT Hi = &L.H[(size_t)i * 4 * UL];
      int32_t* HGLN_RESTRICT Hri = Hr + (size_t)i * 4 * UL; const uint32_t* rp = RndMap<UL>::word_p(Rp, i);
      for (int hh = 0; hh < NH; hh++) {
        const __m512i pv = _mm512_cvtepi16_epi64(_mm_loadu_si128((const __m128i*)(pi + 8 * hh))), cp = _mm512_mul_epi32(pv, si);
        __m512i ssum = _mm512_setzero_si512();
        for (int b = 0; b < 4; b++) {
          const __m512i hv = _mm512_cvtepi16_epi64(_mm_loadu_si128((const __m128i*)(Hi + b * UL + 8 * hh)));
          ssum = _mm512_add_epi64(ssum, _mm512_mul_epi32(cbv[b][hh], hv));
          const __m512i hr = _mm512_add_epi64(_mm512_slli_epi64(hv, 8), rne64_512(_mm512_mul_epi32(cbv[b][hh], cp), Fh));
          n2v[b][hh] = _mm512_add_epi64(n2v[b][hh], _mm512_mul_epi32(hr, hr));
          _mm256_storeu_si256((__m256i*)(Hri + b * UL + 8 * hh), _mm512_cvtepi64_epi32(hr));
        }
        const __m512i Y = _mm512_add_epi64(_mm512_sll_epi64(pv, _mm_cvtsi32_si128(Fp)), _mm512_mullo_epi64(ssum, si));
        const __m512i np_ = clamp64_512(sr64_512(Y, Fp, half8_512(rp + 8 * hh, i & 1)), 0, (int64_t)3 << F.FP);
        _mm_storeu_si128((__m128i*)(pi + 8 * hh), _mm512_cvtepi64_epi16(np_));
      }
    }
#else
    {
      const V64 cb0 = ld<int64_t, UL>(cb[0]), cb1 = ld<int64_t, UL>(cb[1]), cb2 = ld<int64_t, UL>(cb[2]), cb3 = ld<int64_t, UL>(cb[3]);
      V64 n0 = splat<int64_t, UL>(0), n1 = n0, n2 = n0, n3 = n0;
      for (int i = 0; i < D; i++) {
        const int64_t si = s[i]; int16_t* HGLN_RESTRICT pi = &L.p[(size_t)i * UL]; const int16_t* HGLN_RESTRICT Hi = &L.H[(size_t)i * 4 * UL];
        int32_t* HGLN_RESTRICT Hri = Hr + (size_t)i * 4 * UL;
        const V64 pv = cvt<V64>(ld<int16_t, UL>(pi)), cp = pv * si;
        const V64 h0 = cvt<V64>(ld<int16_t, UL>(Hi)), h1 = cvt<V64>(ld<int16_t, UL>(Hi + UL)), h2 = cvt<V64>(ld<int16_t, UL>(Hi + 2 * UL)), h3 = cvt<V64>(ld<int16_t, UL>(Hi + 3 * UL));
        const V64 ssum = cb0 * h0 + cb1 * h1 + cb2 * h2 + cb3 * h3;
        const V64 r0 = (h0 << 8) + vrne<int64_t, UL>(cb0 * cp, Fh), r1 = (h1 << 8) + vrne<int64_t, UL>(cb1 * cp, Fh);
        const V64 r2 = (h2 << 8) + vrne<int64_t, UL>(cb2 * cp, Fh), r3 = (h3 << 8) + vrne<int64_t, UL>(cb3 * cp, Fh);
        n0 += r0 * r0; n1 += r1 * r1; n2 += r2 * r2; n3 += r3 * r3;
        st<int32_t, UL>(Hri, cvt<V32>(r0)); st<int32_t, UL>(Hri + UL, cvt<V32>(r1)); st<int32_t, UL>(Hri + 2 * UL, cvt<V32>(r2)); st<int32_t, UL>(Hri + 3 * UL, cvt<V32>(r3));
        const V64 Y = (pv << Fp) + ssum * si;
        st<int16_t, UL>(pi, cvt<IV<int16_t, UL>>(vclamp<int64_t, UL>(vsr<UL>(Y, Fp, half_lanes<UL>(RndMap<UL>::word_p(Rp, i), i & 1)), 0, (int64_t)3 << F.FP)));
      }
      st<int64_t, UL>(N2[0], n0); st<int64_t, UL>(N2[1], n1); st<int64_t, UL>(N2[2], n2); st<int64_t, UL>(N2[3], n3);
    }
#endif
#if defined(HGLN_Q_AVX512)
    // rc = floor(2^54 / isqrt(N2)), exact: double estimates corrected in integer arithmetic (0 where N2 = 0)
    for (int b = 0; b < 4; b++) for (int hh = 0; hh < NH; hh++) {
      const __m512i n2 = n2v[b][hh], one = _mm512_set1_epi64(1);
      __m512i g = _mm512_cvttpd_epi64(_mm512_sqrt_pd(_mm512_cvtepi64_pd(n2)));
      for (int it = 0; it < 2; it++) {
        g = _mm512_mask_sub_epi64(g, _mm512_cmpgt_epi64_mask(_mm512_mullo_epi64(g, g), n2), g, one);
        const __m512i g1 = _mm512_add_epi64(g, one);
        g = _mm512_mask_add_epi64(g, _mm512_cmple_epi64_mask(_mm512_mullo_epi64(g1, g1), n2), g, one);
      }
      const __mmask8 nz = _mm512_cmpneq_epi64_mask(g, _mm512_setzero_si512());
      const __m512i gs = _mm512_mask_blend_epi64(nz, one, g), T = _mm512_set1_epi64((int64_t)1 << 54);
      __m512i q = _mm512_cvttpd_epi64(_mm512_div_pd(_mm512_set1_pd(18014398509481984.0), _mm512_cvtepi64_pd(gs)));
      __m512i r = _mm512_sub_epi64(T, _mm512_mullo_epi64(q, gs));
      for (int it = 0; it < 2; it++) {
        const __mmask8 neg = _mm512_cmplt_epi64_mask(r, _mm512_setzero_si512());
        q = _mm512_mask_sub_epi64(q, neg, q, one); r = _mm512_mask_add_epi64(r, neg, r, gs);
        const __mmask8 big = _mm512_cmpge_epi64_mask(r, gs);
        q = _mm512_mask_add_epi64(q, big, q, one); r = _mm512_mask_sub_epi64(r, big, r, gs);
      }
      _mm512_store_si512((void*)(rc[b] + 8 * hh), _mm512_maskz_mov_epi64(nz, q));
    }
#else
    for (int b = 0; b < 4; b++) for (int k = 0; k < UL; k++) {
      int64_t rem; rc[b][k] = N2[b][k] ? qs::div_floor((int64_t)1 << 54, (int64_t)qs::isqrt((uint64_t)N2[b][k]), &rem) : 0;
    }
#endif
#if defined(HGLN_Q_AVX512)
    __m512i rcv[4][NH]; __mmask8 keep[4][NH];
    for (int b = 0; b < 4; b++) for (int hh = 0; hh < NH; hh++) { rcv[b][hh] = _mm512_load_si512((const void*)(rc[b] + 8 * hh)); keep[b][hh] = _mm512_cmpeq_epi64_mask(rcv[b][hh], _mm512_setzero_si512()); }
    for (int i = 0; i < D; i++) {
      int16_t* HGLN_RESTRICT Hi = &L.H[(size_t)i * 4 * UL]; const int32_t* HGLN_RESTRICT Hri = Hr + (size_t)i * 4 * UL;
      for (int b = 0; b < 4; b++) {
        const uint32_t* rh = RndMap<UL>::word_h(Rh, i, b);
        for (int hh = 0; hh < NH; hh++) {
          const __m512i hr = _mm512_cvtepi32_epi64(_mm256_loadu_si256((const __m256i*)(Hri + b * UL + 8 * hh)));
          __m512i nh = clamp64_512(sr64_512(_mm512_mullo_epi64(hr, rcv[b][hh]), Fr, half8_512(rh + 8 * hh, i & 1)), -F.MW, F.MW);
          nh = _mm512_mask_blend_epi64(keep[b][hh], nh, _mm512_cvtepi16_epi64(_mm_loadu_si128((const __m128i*)(Hi + b * UL + 8 * hh))));
          _mm_storeu_si128((__m128i*)(Hi + b * UL + 8 * hh), _mm512_cvtepi64_epi16(nh));
        }
      }
    }
#else
    V64 rcv[4]; IV<int64_t, UL> keep[4];
    for (int b = 0; b < 4; b++) { rcv[b] = ld<int64_t, UL>(rc[b]); keep[b] = rcv[b] == splat<int64_t, UL>(0); }
    for (int i = 0; i < D; i++) {
      int16_t* HGLN_RESTRICT Hi = &L.H[(size_t)i * 4 * UL]; const int32_t* HGLN_RESTRICT Hri = Hr + (size_t)i * 4 * UL;
      for (int b = 0; b < 4; b++) {
        const V64 nh = vclamp<int64_t, UL>(vsr<UL>(cvt<V64>(ld<int32_t, UL>(Hri + b * UL)) * rcv[b], Fr, half_lanes<UL>(RndMap<UL>::word_h(Rh, i, b), i & 1)), -F.MW, F.MW);
        st<int16_t, UL>(Hi + b * UL, cvt<IV<int16_t, UL>>(vsel<int64_t, UL>(keep[b], cvt<V64>(ld<int16_t, UL>(Hi + b * UL)), nh)));
      }
    }
#endif
  }
  // rule step (normalised LMS), per unit over its active columns; random word of (slot q, corner c) = word (q>>1)&3,
  // half q&1 of the unit's group q>>3, lane c: the 32 halves of a word row are the 32 int16 of a W slot pair.
  const int Fw = 26 + F.AB - F.WB;
  const int32_t* HGLN_RESTRICT x = w.x.data(); const int32_t* HGLN_RESTRICT h = w.h.data();
  // per unit: its rule steps sc (only the selected corners are non-zero), then the Philox blocks of exactly those
  // corners (all units together, 16 blocks per group); they are scattered to [unit batch][word][corner] - the words of
  // unselected corners are never used (their update is 0, which rounds to itself for any random number)
  alignas(64) int32_t scu[16][32]; int wg0[16], gw = 0, ne = 0;
  uint32_t* HGLN_RESTRICT ce = c0;                                // compressed counters
  uint16_t* HGLN_RESTRICT ent = reinterpret_cast<uint16_t*>(S.scratch);   // (unit batch group, corner) per entry; Hr is dead now
  for (int k = 0; k < U; k++) {
    int64_t h2 = 0;
    for (int c = 0; c < 16; c++) h2 += (int64_t)h[c * UL + k] * h[c * UL + k];
    const int64_t den = qs::rne(h2, 2 * F.FA - 18) * qs::rne(w.x2[k], 2 * F.FA - 12) + 1073742;
    const int64_t c2 = qs::sat32(qs::div_rne((S.Mq * S.es[k]) * ((int64_t)1 << (38 - F.FA)), den));
    const int nb = ((L.ncol[k] + 1) / 2 + 3) / 4;
    wg0[k] = gw; gw += nb;
    for (int c = 0; c < 16; c++) {
      const int32_t v = qs::sat32(qs::rne(c2 * h[c * UL + k], F.FA)); scu[k][2 * c] = scu[k][2 * c + 1] = v;
      if (v) for (int B = 0; B < nb; B++) { ce[ne] = (uint32_t)((k << 16) | (B << 4) | c); ent[2 * ne] = (uint16_t)(wg0[k] + B); ent[2 * ne + 1] = (uint16_t)c; ne++; }
    }
  }
  const int nge = (ne + 15) / 16;
  for (int e = ne; e < 16 * nge; e++) ce[e] = 0;
  uint32_t* HGLN_RESTRICT Rc = R + 64 * (gp + gh);                // compressed output
  uint32_t* HGLN_RESTRICT Rw = Rc + 64 * nge;                     // [unit batch][word][corner]
  philox_groups<P>(ce, nge, 4 * S.layer + qs::kArrW, S.tl, S.th, S.k0, S.k1, Rc);
  for (int e = 0; e < ne; e++) {
    const uint32_t* src = Rc + 64 * (e >> 4) + (e & 15); uint32_t* dst = Rw + 64 * ent[2 * e] + ent[2 * e + 1];
    dst[0] = src[0]; dst[16] = src[16]; dst[32] = src[32]; dst[48] = src[48];
  }
  for (int k = 0; k < U; k++) {
    const int32_t* sc2 = scu[k]; bool any = false;
    for (int c = 0; c < 16; c++) any |= sc2[2 * c] != 0;
    if (!any) continue;
    const int nc = L.ncol[k], np = (nc + 1) / 2; const int32_t* HGLN_RESTRICT ck = &L.cols[(size_t)k * L.cs];
    int16_t* HGLN_RESTRICT Wk = &L.W[(size_t)k * L.cp * 32];
    const uint32_t* Rk = Rw + 64 * wg0[k];
#if defined(HGLN_Q_AVX512)
    __m512i s64[4];
    for (int qq = 0; qq < 4; qq++) s64[qq] = _mm512_cvtepi32_epi64(_mm256_load_si256((const __m256i*)(sc2 + 8 * qq)));
    const __m128i cF = _mm_cvtsi32_si128(Fw), cF16 = _mm_cvtsi32_si128(Fw - 16);
    const __m512i m16 = _mm512_set1_epi64(0xFFFF), one32 = _mm512_set1_epi32(1), lo32 = _mm512_set1_epi32(-F.MW), hi32 = _mm512_set1_epi32(F.MW);
    for (int q2 = 0; q2 < np; q2++) {
      const uint32_t* rw = Rk + 64 * (q2 >> 2) + 16 * (q2 & 3);
      const int64_t x0 = x[ck[2 * q2]], x1 = 2 * q2 + 1 < nc ? x[ck[2 * q2 + 1]] : 0;
      const __m512i xv = _mm512_mask_blend_epi64(0xAA, _mm512_set1_epi64(x0), _mm512_set1_epi64(x1));
      int16_t* HGLN_RESTRICT Wq = Wk + q2 * 32;
      for (int hh = 0; hh < 2; hh++) {                          // elements 16 hh .. 16 hh + 15 (corners 8 hh .. 8 hh + 7)
        const __m512i d0 = _mm512_mul_epi32(s64[2 * hh], xv), d1 = _mm512_mul_epi32(s64[2 * hh + 1], xv);
        const __m512i dq = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi64_epi32(_mm512_sra_epi64(d0, cF))), _mm512_cvtepi64_epi32(_mm512_sra_epi64(d1, cF)), 1);
        const __m512i fr = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi64_epi32(_mm512_and_si512(_mm512_srl_epi64(d0, cF16), m16))),
                                              _mm512_cvtepi64_epi32(_mm512_and_si512(_mm512_srl_epi64(d1, cF16), m16)), 1);
        const __m512i r16 = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(rw + 8 * hh)));
        __m512i res = _mm512_add_epi32(_mm512_cvtepi16_epi32(_mm256_load_si256((const __m256i*)(Wq + 16 * hh))), dq);
        res = _mm512_mask_add_epi32(res, _mm512_cmplt_epi32_mask(r16, fr), res, one32);
        _mm256_store_si256((__m256i*)(Wq + 16 * hh), _mm512_cvtepi32_epi16(_mm512_min_epi32(_mm512_max_epi32(res, lo32), hi32)));
      }
    }
#else
    typedef IV<int64_t, 16> W64;
    const W64 sA = cvt<W64>(ld<int32_t, 16>(sc2)), sB = cvt<W64>(ld<int32_t, 16>(sc2 + 16));
    const W64 odd = cvt<W64>(IV<int32_t, 16>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}) == splat<int64_t, 16>(1);
    for (int q2 = 0; q2 < np; q2++) {
      const uint32_t* rw = Rk + 64 * (q2 >> 2) + 16 * (q2 & 3);
      const int64_t x0 = x[ck[2 * q2]], x1 = 2 * q2 + 1 < nc ? x[ck[2 * q2 + 1]] : 0;
      const W64 xv = vsel<int64_t, 16>(odd, splat<int64_t, 16>(x1), splat<int64_t, 16>(x0));
      int16_t* HGLN_RESTRICT Wq = Wk + q2 * 32;
      alignas(64) uint32_t rr[32];
      for (int c = 0; c < 16; c++) { rr[2 * c] = rw[c] & 0xFFFFu; rr[2 * c + 1] = rw[c] >> 16; }
      const W64 YA = (cvt<W64>(ld<int16_t, 16>(Wq)) << Fw) + sA * xv, YB = (cvt<W64>(ld<int16_t, 16>(Wq + 16)) << Fw) + sB * xv;
      st<int16_t, 16>(Wq, cvt<IV<int16_t, 16>>(vclamp<int64_t, 16>(vsr<16>(YA, Fw, cvt<W64>(ld<uint32_t, 16>(rr))), -F.MW, F.MW)));
      st<int16_t, 16>(Wq + 16, cvt<IV<int16_t, 16>>(vclamp<int64_t, 16>(vsr<16>(YB, Fw, cvt<W64>(ld<uint32_t, 16>(rr + 16))), -F.MW, F.MW)));
    }
#endif
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
    QStep S; S.Mq = qspec::round_param(P.mu, 16); S.tl = (uint32_t)t; S.th = (uint32_t)((uint64_t)t >> 32);
    S.k0 = (uint32_t)seed; S.k1 = (uint32_t)(seed >> 32); S.scratch = scr_.data(); S.c0 = c0_.data(); S.rnd = rnd_.data();
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
  R step(const R* x, R y) { R p = predict(x); learn(y); return p; }
  size_t model_bytes() const { return L[0].model_bytes() + L[1].model_bytes() + L[2].model_bytes() + sizeof(*this); }

 private:
  std::vector<int32_t> scr_; std::vector<uint32_t> c0_, rnd_;
  void scratch_() {                                          // Hr, and the Philox groups of the largest layer step
    const int D = L[0].D, nb = (D + 7) / 8; size_t g = (size_t)5 * nb + 2;
    for (auto& l : L) { size_t gw = 0; for (int k = 0; k < l.U; k++) gw += ((l.ncol[k] + 1) / 2 + 3) / 4; g = std::max(g, (size_t)5 * nb + 2 + 18 * gw); }
    const size_t n = std::max((size_t)D * 4 * L[0].UL, 16 * g);
    if (scr_.size() < n) scr_.assign(n, 0);
    if (c0_.size() < 16 * g) { c0_.assign(16 * g, 0); rnd_.assign(64 * g + 64 * 4, 0); }
  }
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
