// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln.hpp - Hypercube Gated Linear Network (HGLN v5.0; Params::v43() / Params::v42() for the earlier versions). Header-only C++17, no dependencies beyond the standard library.
// Stateless online learner: call predict(x) then learn(y) once per sample.
// Copyright (C) Odin Loch. Licensed under the GNU Affero General Public License v3.0 or later (see LICENSE).
//
// Implementation notes (version 4.2-opt; details and measurements in docs/OPTIMISATION.md). The algorithm is the
// reference procedure (docs/TGN_impl_spec.md section 4); the data layout is built for SIMD:
//   * Per-unit work (gates, heatmap, mixture, competence, gate gradients) runs for all U units of a layer at once,
//     one unit per vector lane: H is [input][hyperplane][unit], p is [input][unit], m is [corner][unit].
//     U = Params::units (8, 12 or 16) is a run-time value; the kernels are instantiated per U (no padded lanes).
//   * Per sample, the gates of all three layers (they depend only on the input) run first, then the three
//     sequential rule stages (Net::infer); with Params::fanout, layers 2 and 3 both read layer 1.
//   * W is [unit][active column][corner]: the 16 rule dot products of a unit are one 16-wide accumulator.
//   * After the one-off prune the inactive columns are removed from W (about 40% less work and memory), and the
//     activity trace A, which only feeds the prune, is freed.
//   * pu = p*u is recomputed instead of stored; the p step and the hyperplane step are fused into one pass.
//   * Each lane keeps the reference's order of operations, so float64 results match it to about 1e-15. Two deliberate
//     1-ulp-level differences: the gate's exp (hgln_gate.h) and renormalising by multiplying with 1/|H|.
//   * No allocation per sample: buffers are allocated in init(), and once more at the prune to shrink W.
//   * infer() is const (caller-owned Work), so many threads can predict at once; independent networks train in
//     parallel (Classifier::train_batch, run_streams, Classifier::set_threads).
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#include "hgln_gate.h"
#if defined(HGLN_WITH_FORGE) || defined(HGLN_SIMD_KERNEL_TU)
#include "hgln_simd.hpp"   // [SIMD hook] run-time dispatched per-CPU kernels (CMake build with FORGE; see src/simd/)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define HGLN_RESTRICT __restrict__
#define HGLN_INLINE inline __attribute__((always_inline))
#define HGLN_NOINLINE __attribute__((noinline))
#else
#define HGLN_NOINLINE
#define HGLN_RESTRICT
#define HGLN_INLINE inline
#endif

namespace hgln {

template <class T> static inline T clipv(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Small deterministic generator (xorshift64) so results do not depend on the platform's <random>.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed = 88172645463325252ULL) : s(seed ? seed : 88172645463325252ULL) { for (int i = 0; i < 8; i++) next(); }
  uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
  double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() { double a = uni() + 1e-300, b = uni(); return std::sqrt(-2 * std::log(a)) * std::cos(6.283185307179586 * b); }
};

namespace detail {
// 64-byte aligned, copyable array. Used for every model and scratch buffer so SIMD loads are aligned.
template <class T> class ABuf {
  T* p_ = nullptr; size_t n_ = 0;
  static T* alloc(size_t n) { return n ? static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(64))) : nullptr; }
  static void release(T* p) { if (p) ::operator delete(p, std::align_val_t(64)); }
 public:
  ABuf() = default;
  ABuf(const ABuf& o) : p_(alloc(o.n_)), n_(o.n_) { if (n_) std::memcpy(p_, o.p_, n_ * sizeof(T)); }
  ABuf(ABuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  ABuf& operator=(const ABuf& o) { if (this != &o) { ABuf t(o); swap(t); } return *this; }
  ABuf& operator=(ABuf&& o) noexcept { if (this != &o) { release(p_); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; } return *this; }
  ~ABuf() { release(p_); }
  void swap(ABuf& o) noexcept { std::swap(p_, o.p_); std::swap(n_, o.n_); }
  void assign(size_t n, T v) { if (n != n_) { release(p_); p_ = alloc(n); n_ = n; } for (size_t i = 0; i < n; i++) p_[i] = v; }
  void reset() { release(p_); p_ = nullptr; n_ = 0; }
  T* data() { return p_; }
  const T* data() const { return p_; }
  size_t size() const { return n_; }
  T* begin() { return p_; }
  T* end() { return p_ + n_; }
  const T* begin() const { return p_; }
  const T* end() const { return p_ + n_; }
  T& operator[](size_t i) { return p_[i]; }
  const T& operator[](size_t i) const { return p_[i]; }
};
}  // namespace detail

// ---------------------------------------------------------------------------------------------------------------------
// Threading helper: run fn(item, worker) for item in [0, n) on up to `threads` threads (0 = all hardware threads).
// Items are handed out dynamically. The calling thread works too. Exceptions must not escape fn.
inline int hardware_threads() { unsigned n = std::thread::hardware_concurrency(); return n ? (int)n : 1; }
template <class F> void parallel_for(int n, int threads, F&& fn) {
  if (n <= 0) return;
  int nt = threads <= 0 ? hardware_threads() : threads;
  nt = std::min(nt, n);
  if (nt <= 1) { for (int i = 0; i < n; i++) fn(i, 0); return; }
  std::atomic<int> next(0);
  auto work = [&](int w) { for (int i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n;) fn(i, w); };
  std::vector<std::thread> pool; pool.reserve(nt - 1);
  for (int w = 1; w < nt; w++) pool.emplace_back(work, w);
  work(0);
  for (auto& th : pool) th.join();
}

// Persistent worker pool for per-sample parallelism (spawning threads per call costs more than one sample's work).
// run(n, fn) calls fn(item, worker) for item in [0, n) on the pool and the calling thread, and returns when all are
// done. Workers spin briefly before sleeping, so back-to-back calls (one per sample) wake quickly.
class ThreadPool {
  std::vector<std::thread> th_;
  std::mutex mu_; std::condition_variable cv_, done_;
  const std::function<void(int, int)>* job_ = nullptr;
  int n_ = 0; std::atomic<int> next_{0}; std::atomic<int> busy_{0}; std::atomic<uint64_t> gen_{0}; bool stop_ = false;
  void loop(int w) {
    uint64_t seen = 0;
    for (;;) {
      for (int spin = 0; spin < 20000 && gen_.load(std::memory_order_acquire) == seen; spin++) std::this_thread::yield();
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || gen_.load(std::memory_order_acquire) != seen; });
        if (stop_) return;
        seen = gen_.load(std::memory_order_acquire);
      }
      for (int i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < n_;) (*job_)(i, w);
      if (busy_.fetch_sub(1, std::memory_order_acq_rel) == 1) { std::lock_guard<std::mutex> lk(mu_); done_.notify_one(); }
    }
  }
 public:
  explicit ThreadPool(int threads = 0) {                     // total threads including the caller; 0 = hardware
    int nt = threads <= 0 ? hardware_threads() : threads;
    for (int w = 1; w < nt; w++) th_.emplace_back([this, w] { loop(w); });
  }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ~ThreadPool() { { std::lock_guard<std::mutex> lk(mu_); stop_ = true; } cv_.notify_all(); for (auto& t : th_) t.join(); }
  int size() const { return (int)th_.size() + 1; }
  void run(int n, const std::function<void(int, int)>& fn) {   // not re-entrant; one caller at a time
    if (n <= 0) return;
    if (th_.empty() || n == 1) { for (int i = 0; i < n; i++) fn(i, 0); return; }
    {
      std::lock_guard<std::mutex> lk(mu_);
      job_ = &fn; n_ = n; next_.store(0, std::memory_order_relaxed); busy_.store((int)th_.size(), std::memory_order_relaxed);
      gen_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    for (int i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < n;) fn(i, 0);
    std::unique_lock<std::mutex> lk(mu_);
    done_.wait(lk, [&] { return busy_.load(std::memory_order_acquire) == 0; });
  }
};

template <class T> struct Layer;
template <class T> struct LayerWork;
// Inputs of one layer's learning step (Layer::update).
template <class T> struct UpdateArgs {
  T y = 0, mu = 0, glr = 0, eg = 0;                          // target, rule step, gate rate, global error y - out
  const T* es = nullptr;                                     // U mixed errors (rule steps; gate steps with gate_err 0)
  const T* u = nullptr;                                      // side information (d+1)
  const T* acc_in = nullptr;                                 // 4 x U gate steps routed here by upper layers, or null
  T* acc_out = nullptr;                                      // 4 x U latch credit routed to the lower layer, or null
  const LayerWork<T>* lo = nullptr;                          // the lower layer's work (latch credit: its kap)
};

// Per-sample scratch of one layer: written by forward, read by update. Not part of the model.
template <class T> struct LayerWork {
  detail::ABuf<T> inp, a, zn, h, f, pred, comp, x2, pp;      // inp: N1 inputs incl. bias; a, zn: 4 x U; h, f: 16 x U
  detail::ABuf<T> kap, hold;                                 // 4 x U: d a / d(12 z) and the latch's held part (Params::latch)
  detail::ABuf<T> hs, sel;                                   // top-K corners (Params::corners < 16): the soft heatmap and
                                                             // the selection (1 / 0), 16 x U; h is then renormalised
  void init(int N1, int U) {
    inp.assign(N1, 0); a.assign(U * 4, 0); zn.assign(U * 4, 0); h.assign(U * 16, 0); f.assign(U * 16, 0);
    kap.assign(U * 4, 0); hold.assign(U * 4, 0); hs.assign(U * 16, 0); sel.assign(U * 16, 0);
    pred.assign(U, 0); comp.assign(U, 0); x2.assign(U, 0); pp.assign(U, 0);
  }
};

// Per-sample scratch of a whole network. Net owns one for predict()/learn(); callers may own more for infer().
template <class T> struct Work {
  detail::ABuf<T> u, tmp;                                    // side information (d+1); gather/step temp (max(N1, D))
  LayerWork<T> lw[3];
  detail::ABuf<T> o, r;                                      // the 3U unit predictions and their mixture weights
  T out = 0;
  template <class NetT> void init(const NetT& net) {
    u.assign(net.d + 1, 0); o.assign(3 * net.L[0].U, 0); r.assign(3 * net.L[0].U, 0);
    int mx = net.d + 1;
    for (int l = 0; l < 3; l++) { lw[l].init(net.L[l].N1, net.L[l].U); mx = std::max(mx, net.L[l].N1); }
    tmp.assign(mx, 0);
  }
};

// Wide vector types are only passed between inline functions of this header (and within one kernel TU), so the
// "ABI changes with -mavx..." note does not apply.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpsabi"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
#endif
namespace detail {
// Small fixed-width vectors. With GCC/Clang these are native vector types (lowered to SSE/AVX/AVX-512 or NEON);
// otherwise a plain array with the same element-wise semantics. Element-wise arithmetic keeps each lane's
// operation order identical to the scalar reference, so results are unchanged.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(HGLN_NO_VECTOR_EXT)   // define to force the plain fallback
template <class T, int N> struct VecT { typedef T type __attribute__((vector_size(N * sizeof(T)), aligned(sizeof(T)), __may_alias__)); };
template <class T, int N> using Vec = typename VecT<T, N>::type;
template <class T, int N> HGLN_INLINE Vec<T, N> vload(const T* p) { return *reinterpret_cast<const Vec<T, N>*>(p); }
template <class T, int N> HGLN_INLINE void vstore(T* p, const Vec<T, N>& v) { *reinterpret_cast<Vec<T, N>*>(p) = v; }
template <class T, int N> HGLN_INLINE Vec<T, N> vzero() { return Vec<T, N>{} ; }
#else
template <class T, int N> struct Vec {
  T v[N];
  T& operator[](int i) { return v[i]; }
  const T& operator[](int i) const { return v[i]; }
  Vec& operator+=(const Vec& o) { for (int i = 0; i < N; i++) v[i] += o.v[i]; return *this; }
  Vec& operator/=(const Vec& o) { for (int i = 0; i < N; i++) v[i] /= o.v[i]; return *this; }
  friend Vec operator+(Vec a, const Vec& b) { a += b; return a; }
  friend Vec operator*(const Vec& a, const Vec& b) { Vec r; for (int i = 0; i < N; i++) r.v[i] = a.v[i] * b.v[i]; return r; }
  friend Vec operator*(const Vec& a, T s) { Vec r; for (int i = 0; i < N; i++) r.v[i] = a.v[i] * s; return r; }
  friend Vec operator*(T s, const Vec& a) { Vec r; for (int i = 0; i < N; i++) r.v[i] = s * a.v[i]; return r; }
  friend Vec operator+(const Vec& a, T s) { Vec r; for (int i = 0; i < N; i++) r.v[i] = a.v[i] + s; return r; }
  friend Vec operator+(T s, const Vec& a) { Vec r; for (int i = 0; i < N; i++) r.v[i] = s + a.v[i]; return r; }
  friend Vec operator-(const Vec& a, const Vec& b) { Vec r; for (int i = 0; i < N; i++) r.v[i] = a.v[i] - b.v[i]; return r; }
  friend Vec operator-(T s, const Vec& a) { Vec r; for (int i = 0; i < N; i++) r.v[i] = s - a.v[i]; return r; }
  friend Vec operator/(const Vec& a, const Vec& b) { Vec r; for (int i = 0; i < N; i++) r.v[i] = a.v[i] / b.v[i]; return r; }
};
template <class T, int N> inline Vec<T, N> vload(const T* p) { Vec<T, N> r; for (int i = 0; i < N; i++) r.v[i] = p[i]; return r; }
template <class T, int N> inline void vstore(T* p, const Vec<T, N>& v) { for (int i = 0; i < N; i++) p[i] = v.v[i]; }
template <class T, int N> inline Vec<T, N> vzero() { Vec<T, N> r; for (int i = 0; i < N; i++) r.v[i] = 0; return r; }
#endif
// The 16-lane vector policy the layer kernels are written against (Layer::forward_k / update_k). The SIMD backend
// (src/simd/) instantiates the same kernels once per CPU target with its own policy.
// V/load/store/zero: the 16-lane vectors of the corner-lane work (rule dot products; 16 corners). L<N>: N-lane
// vectors of the unit-lane work (N = the lane block VB of UnitOps below).
template <class T> struct VxOps {
  typedef Vec<T, 16> V;
  static HGLN_INLINE V load(const T* p) { return vload<T, 16>(p); }
  static HGLN_INLINE void store(T* p, const V& v) { vstore<T, 16>(p, v); }
  static HGLN_INLINE V zero() { return vzero<T, 16>(); }
  template <int N> struct L {
    typedef T S; typedef Vec<T, N> V;
    static HGLN_INLINE V load(const T* p) { return vload<T, N>(p); }
    static HGLN_INLINE void store(T* p, const V& v) { vstore<T, N>(p, v); }
    static HGLN_INLINE V zero() { return vzero<T, N>(); }
  };
};
// Two vectors side by side (lanes [0, NA) in a, [NA, NA + NB) in b), element-wise arithmetic: each lane's operations
// are unchanged. Used for unit counts that are not a power of two (12 = 8 + 4).
template <class A, class B> struct Pair {
  typedef typename A::S S;
  typename A::V a; typename B::V b;
#define HGLN_PAIR_OP(op)                                                                                              \
  friend HGLN_INLINE Pair operator op(const Pair& x, const Pair& y) { Pair o; o.a = x.a op y.a; o.b = x.b op y.b; return o; } \
  friend HGLN_INLINE Pair operator op(const Pair& x, S s) { Pair o; o.a = x.a op s; o.b = x.b op s; return o; }               \
  friend HGLN_INLINE Pair operator op(S s, const Pair& y) { Pair o; o.a = s op y.a; o.b = s op y.b; return o; }               \
  friend HGLN_INLINE Pair& operator op##=(Pair& x, const Pair& y) { x.a = x.a op y.a; x.b = x.b op y.b; return x; }
  HGLN_PAIR_OP(+) HGLN_PAIR_OP(-) HGLN_PAIR_OP(*) HGLN_PAIR_OP(/)
#undef HGLN_PAIR_OP
};
// The unit-lane vectors of a layer with UU units, so that no lane is padding and the accumulators stay in registers:
// a power of two (8, 16) is one policy vector; otherwise the largest power of two below UU plus the rest
// (12 = 8 + 4), processed together (measured: float32 d=64 about 10% faster than three 4-lane blocks, float64 equal).
template <int UU> constexpr int low_pow2(int x = UU) { return x & (x - 1) ? low_pow2<UU>(x & (x - 1)) : x; }   // highest bit
template <class O, int UU, bool P2 = ((UU & (UU - 1)) == 0)> struct UnitOps {
  typedef typename O::template L<UU> B;
  typedef typename B::V V;
  static HGLN_INLINE V load(const typename B::S* p) { return B::load(p); }
  static HGLN_INLINE void store(typename B::S* p, const V& v) { B::store(p, v); }
  static HGLN_INLINE V zero() { return B::zero(); }
};
template <class O, int UU> struct UnitOps<O, UU, false> {
  static constexpr int NA = low_pow2<UU>();
  typedef typename O::template L<NA> A;
  static constexpr int NR = UU - NA;
  typedef UnitOps<O, NR> R;
  struct RB { typedef typename A::S S; typedef typename R::V V; };
  typedef Pair<A, RB> V;
  static HGLN_INLINE V load(const typename A::S* p) { V v; v.a = A::load(p); v.b = R::load(p + NA); return v; }
  static HGLN_INLINE void store(typename A::S* p, const V& v) { A::store(p, v.a); R::store(p + NA, v.b); }
  static HGLN_INLINE V zero() { V v; v.a = A::zero(); v.b = R::zero(); return v; }
};
}  // namespace detail

// Units per layer (Params::units): one of these. Layers 2 and 3 take the previous layer's U outputs as inputs.
constexpr int kUnitChoices[3] = {8, 12, 16};
constexpr int kMaxUnits = 16;
inline bool valid_units(int u) { return u == 8 || u == 12 || u == 16; }
inline int unit_index(int u) { return u == 8 ? 0 : (u == 12 ? 1 : 2); }   // index into the per-U kernel entries

// One layer: U units; each unit = 4 hyperplanes -> 16-corner heatmap -> 16 linear rules blended by the heatmap.
template <class T> struct Layer {
  static constexpr T BETA = 12;
  int U = 16;                                                // units per layer (8, 12 or 16; Params::units)
  int n = 0, D = 0, N1 = 0;                                  // inputs, side-info size (d+1), inputs+bias
  int nt = 0;                                                // prune target: active inputs kept per unit
  int cs = 0;                                                // column stride of W / cols (max active columns per unit)
  bool pruned = false;                                       // the one-off prune has run (A is then freed)
  bool has_competence = false;                               // keep the competence map m (version 4.2 mixture)
  T latch = 0;                                               // cross-layer latch theta (Params::latch; layers 2 and 3)
  T credit = 0;                                              // latch credit scale (Params::latch_credit)
  int gate_err = 0;                                          // Params::gate_err: 0 mixed es, 1 own error, 2 global error
  int corners = 16;                                          // Params::corners: top-K corners per unit (16 = all)
  detail::ABuf<T> H;                                         // D x 4 x U   hyperplanes, NOT normalised: [i][b][unit]
  detail::ABuf<T> q, nq, rs;                                 // 4 x U       |H_b|^2, |H_b| and 1/|H_b| (norm tracking)
  detail::ABuf<T> p;                                         // D x U       gate weights: [i][unit]
  detail::ABuf<T> W;                                         // U x cs x 16 rules, corner innermost; active columns only
  detail::ABuf<T> m;                                         // 16 x U      competence map: [corner][unit] (if competence)
  detail::ABuf<T> A;                                         // U x N1      input activity (only until the prune)
  detail::ABuf<int32_t> cols;                                // U x cs      active column index (ascending, bias last)
  detail::ABuf<int32_t> ncol;                                // U           number of active columns incl. bias

  void init(int n_, int D_, bool competence_ = false, int units = 16) {
    if (!valid_units(units)) throw std::invalid_argument("hgln: units per layer must be 8, 12 or 16");
    U = units; n = n_; D = D_; N1 = n + 1; cs = N1; pruned = false; has_competence = competence_;
    nt = n <= 4 ? n : (int)std::nearbyint(0.6 * n);
    H.assign((size_t)D * U * 4, 0); p.assign((size_t)D * U, 1); W.assign((size_t)U * cs * 16, 0);
    q.assign(4 * U, 1); nq.assign(4 * U, 1); rs.assign(4 * U, 1);
    if (has_competence) m.assign(U * 16, (T)0.3); else m.reset();
    A.assign((size_t)U * N1, (T)1e-3);
    cols.assign((size_t)U * cs, 0); ncol.assign(U, N1);
    for (int k = 0; k < U; k++) for (int j = 0; j < N1; j++) cols[k * cs + j] = j;
  }
  size_t hidx(int k, int b, int i) const { return ((size_t)i * 4 + b) * U + k; }
  void random_hyperplanes(Rng& rng) {                        // unit-norm rows, last (bias) component 0
    std::vector<double> v(D);
    for (int k = 0; k < U; k++) for (int b = 0; b < 4; b++) {
      double nn = 0;
      for (int i = 0; i < D - 1; i++) { v[i] = rng.normal(); nn += v[i] * v[i]; }
      v[D - 1] = 0; nn = std::sqrt(nn);
      for (int i = 0; i < D; i++) H[hidx(k, b, i)] = (T)(v[i] / nn);
    }
    recompute_norms();
  }
  // Exact |H_b| from the stored (unnormalised) hyperplanes.
  void recompute_norms() {
    for (int b = 0; b < 4; b++) for (int k = 0; k < U; k++) {
      T s2 = 0; for (int i = 0; i < D; i++) { T v = H[hidx(k, b, i)]; s2 += v * v; }
      q[b * U + k] = s2; nq[b * U + k] = std::sqrt(s2); rs[b * U + k] = (T)1 / nq[b * U + k];
    }
  }
  // Scale every hyperplane back to unit length (exactly the same model; bounds rounding drift in the tracked norms).
  void renormalise() {
    for (int i = 0; i < D; i++) for (int b = 0; b < 4; b++) for (int k = 0; k < U; k++) H[hidx(k, b, i)] *= rs[b * U + k];
    recompute_norms();
  }
  // ---- accessors in the reference (dense) layout: H as U x 4 x D, p as U x D, W as U x 16 x N1, M as U x N1
  // set_hyperplanes takes any non-zero rows (the model uses their directions); get_hyperplanes/hyperplane return unit
  // rows, as the reference stores them. The *_raw forms and hyperplane_sqnorm expose the stored state exactly.
  template <class S> void set_hyperplanes(const S* src, const S* sqnorms = nullptr) {
    for (int k = 0; k < U; k++) for (int b = 0; b < 4; b++) for (int i = 0; i < D; i++) H[hidx(k, b, i)] = (T)src[((size_t)k * 4 + b) * D + i];
    recompute_norms();
    if (sqnorms) for (int k = 0; k < U; k++) for (int b = 0; b < 4; b++) {
      T s2 = (T)sqnorms[(size_t)k * 4 + b]; q[b * U + k] = s2; nq[b * U + k] = std::sqrt(s2); rs[b * U + k] = (T)1 / nq[b * U + k];
    }
  }
  template <class S> void get_hyperplanes(S* dst) const {
    for (int k = 0; k < U; k++) for (int b = 0; b < 4; b++) for (int i = 0; i < D; i++) dst[((size_t)k * 4 + b) * D + i] = (S)hyperplane(k, b, i);
  }
  template <class S> void get_hyperplanes_raw(S* dst, S* sqnorms) const {
    for (int k = 0; k < U; k++) for (int b = 0; b < 4; b++) {
      for (int i = 0; i < D; i++) dst[((size_t)k * 4 + b) * D + i] = (S)H[hidx(k, b, i)];
      sqnorms[(size_t)k * 4 + b] = (S)q[b * U + k];
    }
  }
  T hyperplane(int k, int b, int i) const { return H[hidx(k, b, i)] * rs[b * U + k]; }
  T hyperplane_raw(int k, int b, int i) const { return H[hidx(k, b, i)]; }
  T hyperplane_sqnorm(int k, int b) const { return q[b * U + k]; }
  T gate_weight(int k, int i) const { return p[(size_t)i * U + k]; }
  void set_gate_weight(int k, int i, T v) { p[(size_t)i * U + k] = v; }
  int slot(int k, int j) const {                             // position of input column j in unit k's compact W, or -1
    const int32_t* ck = &cols[(size_t)k * cs];
    for (int q = 0; q < ncol[k]; q++) if (ck[q] == j) return q;
    return -1;
  }
  bool active(int k, int j) const { return slot(k, j) >= 0; }  // the reference's M[k][j] > 0
  T rule(int k, int c, int j) const { int q = slot(k, j); return q < 0 ? (T)0 : W[((size_t)k * cs + q) * 16 + c]; }  // 0 if pruned
  T competence(int k, int c) const { return has_competence ? m[(size_t)c * U + k] : (T)0; }
  T activity(int k, int j) const { return pruned ? (T)0 : A[(size_t)k * N1 + j]; }
  // Replace the whole layer state from the reference (dense) layout: Hd U x 4 x D, pd U x D, Wd U x 16 x N1,
  // Md U x N1 (0/1 mask; bias column must be 1), Ad U x N1 (ignored when is_pruned), md U x 16 (ignored without
  // competence), qd U x 4 squared hyperplane norms (nullptr: computed from Hd). Inverse of the accessors above; used to
  // bring state back from the GPU kernel's flat layout.
  template <class S> void import_dense(const S* Hd, const S* pd, const S* Wd, const S* Md, const S* Ad, const S* md, bool is_pruned, const S* qd = nullptr) {
    set_hyperplanes(Hd, qd);
    for (int k = 0; k < U; k++) {
      for (int i = 0; i < D; i++) p[(size_t)i * U + k] = (T)pd[(size_t)k * D + i];
      if (has_competence && md) for (int c = 0; c < 16; c++) m[(size_t)c * U + k] = (T)md[(size_t)k * 16 + c];
    }
    int newcs = 0;
    for (int k = 0; k < U; k++) { int c = 0; for (int j = 0; j < N1; j++) c += Md[(size_t)k * N1 + j] > 0 || j == n; newcs = std::max(newcs, c); }
    cs = newcs; W.assign((size_t)U * cs * 16, 0); cols.assign((size_t)U * cs, 0);
    for (int k = 0; k < U; k++) {
      int q = 0;
      for (int j = 0; j < N1; j++) {
        if (!(Md[(size_t)k * N1 + j] > 0 || j == n)) continue;
        cols[(size_t)k * cs + q] = j;
        for (int c = 0; c < 16; c++) W[((size_t)k * cs + q) * 16 + c] = (T)Wd[((size_t)k * 16 + c) * N1 + j];
        q++;
      }
      ncol[k] = q;
    }
    pruned = is_pruned;
    if (pruned) A.reset();
    else { A.assign((size_t)U * N1, 0); for (size_t i = 0; i < (size_t)U * N1; i++) A[i] = (T)Ad[i]; }
  }
  size_t model_bytes() const {
    return (H.size() + q.size() + nq.size() + rs.size() + p.size() + W.size() + m.size() + A.size()) * sizeof(T) + (cols.size() + ncol.size()) * sizeof(int32_t);
  }

  // ---- [SIMD hook] With HGLN_WITH_FORGE, gates/rules/update call the kernels compiled for this CPU (hgln_simd.hpp,
  // chosen once at run time); otherwise, and inside the kernel TUs, they are the inline kernels below. Either way
  // the kernel is picked once per call by U (template argument UU = U).
  // forward = gates (depend only on the side information u) then rules (depend on the previous layer's outputs).
#if defined(HGLN_WITH_FORGE) && !defined(HGLN_SIMD_KERNEL_TU)
  void gates(const T* u, LayerWork<T>& w, const LayerWork<T>* lo = nullptr) const { simd::layer_kernels<T>().gates[unit_index(U)](*this, u, w, lo); }
  void rules(const T* in, LayerWork<T>& w, T* xg) const { simd::layer_kernels<T>().rules[unit_index(U)](*this, in, w, xg); }
  void update(const UpdateArgs<T>& a, const LayerWork<T>& w) { simd::layer_kernels<T>().update[unit_index(U)](*this, a, w); }
#else
  void gates(const T* u, LayerWork<T>& w, const LayerWork<T>* lo = nullptr) const {
    switch (U) { case 8: gates_k<detail::VxOps<T>, 8>(u, w, lo); break; case 12: gates_k<detail::VxOps<T>, 12>(u, w, lo); break; default: gates_k<detail::VxOps<T>, 16>(u, w, lo); }
  }
  void rules(const T* in, LayerWork<T>& w, T* xg) const {
    switch (U) { case 8: rules_k<detail::VxOps<T>, 8>(in, w, xg); break; case 12: rules_k<detail::VxOps<T>, 12>(in, w, xg); break; default: rules_k<detail::VxOps<T>, 16>(in, w, xg); }
  }
  void update(const UpdateArgs<T>& a, const LayerWork<T>& w) {
    switch (U) {
      case 8: update_k<detail::VxOps<T>, 8>(a, w); break;
      case 12: update_k<detail::VxOps<T>, 12>(a, w); break;
      default: update_k<detail::VxOps<T>, 16>(a, w);
    }
  }
#endif
  // compatibility form of update (no latch credit; global error unused unless gate_err = 2)
  void update(T y, T mu, T glr, const T* es, const T* u, const LayerWork<T>& w, T eg = 0) {
    UpdateArgs<T> a; a.y = y; a.mu = mu; a.glr = glr; a.eg = eg; a.es = es; a.u = u; update(a, w);
  }
  void forward(const T* in, const T* u, LayerWork<T>& w, T* xg, const LayerWork<T>* lo = nullptr) const { gates(u, w, lo); rules(in, w, xg); }

  // Forward pass for all units = gates_k then rules_k. in: n inputs; u: D side-information values; xg: scratch of at
  // least N1. Lane = unit for everything except the rule dot products (lane = corner). O: vector policy (VxOps);
  // UU: compile-time copy of U (unit-lane work runs on detail::UnitOps<O, UU>, corner-lane work on O::V).
  template <class O, int UU> HGLN_INLINE void forward_k(const T* HGLN_RESTRICT in, const T* HGLN_RESTRICT u, LayerWork<T>& w, T* HGLN_RESTRICT xg, const LayerWork<T>* lo = nullptr) const {
    gates_k<O, UU>(u, w, lo); rules_k<O, UU>(in, w, xg);
  }
  // Gates and heatmap: depend only on u, so all layers' gates can run before any rule stage (Net::infer).
  // lo: the layer whose gate bits this layer's cross-layer latch holds (Params::latch > 0, layers 2 and 3), or null.
  template <class O, int UU> HGLN_NOINLINE void gates_k(const T* HGLN_RESTRICT u, LayerWork<T>& w, const LayerWork<T>* lo = nullptr) const {
    typedef detail::UnitOps<O, UU> Q; typedef typename Q::V VU;
    // gates: z_b = (H_b . (p * u)) / |H_b| with the stored, unnormalised H_b, i ascending; |p * u|^2 for the update
    VU z0 = Q::zero(), z1 = z0, z2 = z0, z3 = z0, pp = z0;
    for (int i = 0; i < D; i++) {
      const T* HGLN_RESTRICT Hi = &H[(size_t)i * 4 * UU];
      const VU pu = Q::load(&p[(size_t)i * UU]) * u[i];
      z0 += Q::load(Hi) * pu; z1 += Q::load(Hi + UU) * pu;
      z2 += Q::load(Hi + 2 * UU) * pu; z3 += Q::load(Hi + 3 * UU) * pu;
      pp += pu * pu;
    }
    gates_finish_k<O, UU>(z0, z1, z2, z3, pp, w, lo);
  }
  template <class O, int UU, class VU> HGLN_INLINE void gates_finish_k(const VU& z0, const VU& z1, const VU& z2, const VU& z3, const VU& pp, LayerWork<T>& w, const LayerWork<T>* lo = nullptr) const {
    typedef detail::UnitOps<O, UU> Q;
    T* HGLN_RESTRICT zz = w.zn.data();
    Q::store(zz, z0 * Q::load(&rs[0])); Q::store(zz + UU, z1 * Q::load(&rs[UU]));
    Q::store(zz + 2 * UU, z2 * Q::load(&rs[2 * UU])); Q::store(zz + 3 * UU, z3 * Q::load(&rs[3 * UU]));
    Q::store(w.pp.data(), pp);
    T* HGLN_RESTRICT aw = w.a.data();
    for (int q = 0; q < 4 * UU; q++) aw[q] = hgln_gate<T>(zz[q]);                 // 4U gates, [b][unit]
    if (latch > 0 && lo) {                // cross-layer latch: set/reset around +-theta, holding the lower layer's bit
      // in simple loops, so that hgln_exp vectorises like the gate (the same scalar operations per gate)
      alignas(64) T es[4 * kMaxUnits], er[4 * kMaxUnits];
      for (int q = 0; q < 4 * UU; q++) { es[q] = -BETA * (zz[q] - latch); er[q] = -BETA * (-zz[q] - latch); }
      for (int q = 0; q < 4 * UU; q++) { es[q] = hgln_exp<T>(es[q]); er[q] = hgln_exp<T>(er[q]); }
      for (int q = 0; q < 4 * UU; q++) {
        const T st = (T)1 / ((T)1 + es[q]), rt = (T)1 / ((T)1 + er[q]), q0 = lo->a[q];
        aw[q] = clipv(st + ((T)1 - st - rt) * q0, (T)1e-4, (T)(1 - 1e-4));
        w.kap[q] = st * ((T)1 - st) * ((T)1 - q0) + rt * ((T)1 - rt) * q0; w.hold[q] = (T)1 - st - rt;
      }
    } else if (latch > 0) {
      for (int q = 0; q < 4 * UU; q++) { w.kap[q] = aw[q] * ((T)1 - aw[q]); w.hold[q] = 0; }
    }
    // 16-corner heatmap h[c] as a tensor product; same multiplication order as the reference (bit 0 first)
    const VU a0 = Q::load(aw), a1 = Q::load(aw + UU), a2 = Q::load(aw + 2 * UU), a3 = Q::load(aw + 3 * UU);
    const VU one = Q::zero() + (T)1;
    const VU b0 = one - a0, b1 = one - a1, b2 = one - a2, b3 = one - a3;
    VU q1[4] = {b0 * b1, a0 * b1, b0 * a1, a0 * a1}, q2[8];
    for (int c = 0; c < 4; c++) { q2[c] = q1[c] * b2; q2[c + 4] = q1[c] * a2; }
    T* HGLN_RESTRICT h = w.h.data();
    for (int c = 0; c < 8; c++) { Q::store(h + c * UU, q2[c] * b3); Q::store(h + (c + 8) * UU, q2[c] * a3); }
    if (corners < 16) top_corners<UU>(w);
  }
  // Top-K corners (Params::corners = K < 16), per unit: corner c is selected iff fewer than K corners rank above it
  // (larger soft value, ties to the lower corner); the soft heatmap and the selection are kept for the gate gradient,
  // and h becomes the selected values renormalised to sum 1 (sum over the selected corners, c ascending), 0 elsewhere.
  // The rule mixture and the rule step use this h, the gate gradient the soft values of the selected corners.
  // Ranks without branches, all units at once (lane = unit): for each pair c < c2 the corner with the larger soft value
  // (ties: the lower corner, c) ranks above the other, which gets + 1. The same ranks as counting per corner. The
  // selection is applied as a factor 1 / 0, which gives the scalar definition's bits (adding + 0 and multiplying by 1
  // change nothing; every soft value is > 0, the gates being clipped to [1e-4, 1 - 1e-4]).
  template <int UU> HGLN_INLINE void top_corners(LayerWork<T>& w) const {   // UU: units, a compile-time constant
    T* HGLN_RESTRICT h = w.h.data(); T* HGLN_RESTRICT hs = w.hs.data(); T* HGLN_RESTRICT sel = w.sel.data();
    for (int q = 0; q < 16 * UU; q++) hs[q] = h[q];
#if (defined(__GNUC__) || defined(__clang__)) && !defined(HGLN_NO_VECTOR_EXT)
    top_block<(UU >= 16 ? 16 : 8)>(hs, h, sel, UU, 0);
    if (UU == 12) top_block<4>(hs, h, sel, UU, 8);
#else
    alignas(64) int32_t rank[16][kMaxUnits]; alignas(64) T sm[kMaxUnits];
    for (int c = 0; c < 16; c++) for (int k = 0; k < UU; k++) rank[c][k] = 0;
    for (int c = 0; c < 16; c++)
      for (int c2 = c + 1; c2 < 16; c2++)
        for (int k = 0; k < UU; k++) { const int32_t gt = hs[c2 * UU + k] > hs[c * UU + k]; rank[c][k] += gt; rank[c2][k] += 1 - gt; }
    for (int k = 0; k < UU; k++) sm[k] = 0;
    for (int c = 0; c < 16; c++) for (int k = 0; k < UU; k++) if (rank[c][k] < corners) sm[k] += hs[c * UU + k];   // c ascending
    for (int c = 0; c < 16; c++)
      for (int k = 0; k < UU; k++) { const bool on = rank[c][k] < corners; sel[c * UU + k] = on ? (T)1 : (T)0; h[c * UU + k] = on ? hs[c * UU + k] / sm[k] : (T)0; }
#endif
  }
#if (defined(__GNUC__) || defined(__clang__)) && !defined(HGLN_NO_VECTOR_EXT)
  // units [k0, k0 + B) of top_corners as one B-lane vector
  template <int B> HGLN_INLINE void top_block(const T* HGLN_RESTRICT hs, T* HGLN_RESTRICT h, T* HGLN_RESTRICT sel, int UU, int k0) const {
    typedef detail::Vec<T, B> VT;
    VT v[16];
    for (int c = 0; c < 16; c++) v[c] = detail::vload<T, B>(hs + c * UU + k0);
    typedef decltype(v[0] > v[0]) VM;                         // -1 / 0 lanes
    VM r[16];
    for (int c = 0; c < 16; c++) r[c] = v[0] != v[0];         // 0 (no NaN)
    for (int c = 0; c < 16; c++)
      for (int c2 = c + 1; c2 < 16; c2++) { const VM gt = v[c2] > v[c]; r[c] -= gt; r[c2] += 1 + gt; }
    VT sm = detail::vzero<T, B>(), sl[16];
    for (int c = 0; c < 16; c++) { sl[c] = (T)0 - __builtin_convertvector(r[c] < corners, VT); sm += v[c] * sl[c]; }   // c ascending
    for (int c = 0; c < 16; c++) { detail::vstore<T, B>(h + c * UU + k0, (v[c] / sm) * sl[c]); detail::vstore<T, B>(sel + c * UU + k0, sl[c]); }
  }
#endif
  // Rule stage: rule outputs, the unit mixture of the 16 rules, the competence and the clipped unit predictions.
  template <class O, int UU> HGLN_NOINLINE void rules_k(const T* HGLN_RESTRICT in, LayerWork<T>& w, T* HGLN_RESTRICT xg) const {
    typedef typename O::V V16; typedef detail::UnitOps<O, UU> Q; typedef typename Q::V VU;
    T* HGLN_RESTRICT inp = w.inp.data();
    for (int j = 0; j < n; j++) inp[j] = in[j];
    inp[n] = 1;
    // rule outputs f[c][unit]: per unit, gather the active inputs and run 16 dot products as one corner-lane accumulator
    T* HGLN_RESTRICT f = w.f.data();
    for (int k = 0; k < UU; k++) {
      const int nc = ncol[k]; const int32_t* HGLN_RESTRICT ck = &cols[(size_t)k * cs];
      T x2 = 0;
      for (int q = 0; q < nc; q++) { T v = inp[ck[q]]; xg[q] = v; x2 += v * v; }
      w.x2[k] = x2;
      const T* HGLN_RESTRICT Wk = &W[(size_t)k * cs * 16];
      V16 acc = O::zero();
      for (int q = 0; q < nc; q++) acc += O::load(Wk + q * 16) * xg[q];
      alignas(64) T fa[16];
      O::store(fa, acc);
      for (int c = 0; c < 16; c++) f[c * UU + k] = fa[c];
    }
    // mixture of the 16 rules (and, with competence, the competence-weighted confidence), all units at once
    const T* HGLN_RESTRICT h = w.h.data();
    VU mix = Q::zero(), cm = mix;
    if (has_competence) {
      for (int c = 0; c < 16; c++) { const VU hc = Q::load(h + c * UU); mix += hc * Q::load(f + c * UU); cm += hc * Q::load(&m[c * UU]); }
      Q::store(w.comp.data(), (Q::zero() + (T)1) / (cm + (T)1e-3));
    } else {
      for (int c = 0; c < 16; c++) mix += Q::load(h + c * UU) * Q::load(f + c * UU);
    }
    alignas(64) T mx[UU];
    Q::store(mx, mix);
    for (int k = 0; k < UU; k++) w.pred[k] = clipv(mx[k], (T)-2, (T)2);
  }

  template <class O, int UU> HGLN_NOINLINE void update_k(const UpdateArgs<T>& args, const LayerWork<T>& w) {
    typedef typename O::V V16; typedef detail::UnitOps<O, UU> Q; typedef typename Q::V VU;
    const T y = args.y, mu = args.mu, glr = args.glr; const T* HGLN_RESTRICT es = args.es; const T* HGLN_RESTRICT u = args.u;
    const T* HGLN_RESTRICT inp = w.inp.data(); const T* HGLN_RESTRICT h = w.h.data(); const T* HGLN_RESTRICT f = w.f.data();
    const VU esv = Q::load(es);
    // competence map: m[c] += 0.02 * h[c] * (e^2 - m[c]), e = the unit's own error
    if (has_competence) {
      const VU e = (Q::zero() + y) - Q::load(w.pred.data()), e2 = e * e;
      for (int c = 0; c < 16; c++) {
        T* HGLN_RESTRICT mc = &m[c * UU];
        const VU mv = Q::load(mc);
        Q::store(mc, mv + (T)0.02 * Q::load(h + c * UU) * (e2 - mv));
      }
    }
    // gate gradient: F1_b = sum over corners with bit b set of h*f, F0_b = tot - F1_b (c ascending, as the reference)
    alignas(64) T cbs[4][UU];                                                     // gate steps [b][unit]
    alignas(64) T c2s[UU];
    {
      VU tot = Q::zero(), F1[4] = {tot, tot, tot, tot}, h2 = tot;
      const bool tk = corners < 16;                          // top-K: gradient from the soft values of the selected corners
      const T* HGLN_RESTRICT hsv = w.hs.data(); const T* HGLN_RESTRICT slv = w.sel.data();
      for (int c = 0; c < 16; c++) {
        const VU hc = Q::load(h + c * UU);
        VU hf;
        if (tk) hf = (Q::load(hsv + c * UU) * Q::load(f + c * UU)) * Q::load(slv + c * UU);
        else hf = hc * Q::load(f + c * UU);
        tot += hf; h2 += hc * hc;
        if (c & 1) F1[0] += hf;
        if (c & 2) F1[1] += hf;
        if (c & 4) F1[2] += hf;
        if (c & 8) F1[3] += hf;
      }
      const T* HGLN_RESTRICT aw = w.a.data();
      // error driving the gate steps (Params::gate_err): mixed es, the unit's own error, or the global error
      alignas(64) T egv[UU];
      if (gate_err == 2) for (int k = 0; k < UU; k++) egv[k] = args.eg;
      else if (gate_err == 1) for (int k = 0; k < UU; k++) egv[k] = y - w.pred[k];
      const VU gev = gate_err == 0 ? esv : Q::load(egv);
      const bool latched = latch > 0 && args.lo;
      for (int b = 0; b < 4; b++) {
        const VU ab = Q::load(aw + b * UU), F0 = tot - F1[b];
        alignas(64) T gv[UU];
        if (latched) {                    // g = 12 kap (F1/a - F0/(1-a)); credit: the held part, through the lower bit
          const VU dm = F1[b] / ab - F0 / ((T)1 - ab);
          Q::store(gv, glr * gev * (BETA * Q::load(&w.kap[b * UU]) * dm));
          if (credit > 0 && args.acc_out) {
            alignas(64) T cv[UU];
            Q::store(cv, glr * gev * (credit * BETA * Q::load(&args.lo->kap[b * UU]) * dm * Q::load(&w.hold[b * UU])));
            for (int k = 0; k < UU; k++) args.acc_out[b * UU + k] += clipv(cv[k], (T)-0.05, (T)0.05);
          }
        } else {
          const VU g = BETA * (F1[b] * ((T)1 - ab) - F0 * ab);
          Q::store(gv, glr * gev * g);
        }
        for (int k = 0; k < UU; k++) cbs[b][k] = clipv(gv[k], (T)-0.05, (T)0.05);
        if (args.acc_in) for (int k = 0; k < UU; k++) cbs[b][k] += args.acc_in[b * UU + k];
      }
      // normalised LMS rate per unit
      Q::store(c2s, (mu * esv) / ((T)1e-3 + h2 * Q::load(w.x2.data())));
    }
    // rule step: W_c += c2 * h_c * xm for every corner, over the active columns only (lane = corner)
    for (int k = 0; k < UU; k++) {
      alignas(64) T sc[16];
      for (int c = 0; c < 16; c++) sc[c] = c2s[k] * h[c * UU + k];
      const V16 scv = O::load(sc);
      const int nc = ncol[k]; const int32_t* HGLN_RESTRICT ck = &cols[(size_t)k * cs];
      T* HGLN_RESTRICT Wk = &W[(size_t)k * cs * 16];
      for (int q = 0; q < nc; q++) {
        T* Wq = Wk + q * 16;
        O::store(Wq, O::load(Wq) + scv * inp[ck[q]]);
      }
      if (!pruned) {                                                                // activity only feeds the prune
        T* HGLN_RESTRICT Ak = &A[(size_t)k * N1];
        for (int j = 0; j < N1; j++) Ak[j] = (T)0.98 * Ak[j] + (T)0.02 * std::fabs(inp[j]);
      }
    }
    // all units at once (unit = lane), fused: p step with the OLD unit hyperplanes H_b/|H_b|, hyperplane step with the
    // OLD p (the forward pass's p*u). The reference adds cb*pu to the unit hyperplane and renormalises; here the stored
    // H_b is unnormalised, so the same direction is H_b += (cb |H_b|) pu, and |H_b|^2 follows in closed form:
    //   |H_b + s pu|^2 = |H_b|^2 + 2 s (H_b . pu) + s^2 |pu|^2,  s = cb |H_b|,  H_b . pu = |H_b| z_b (from the forward pass)
    // so neither a norm pass nor a renormalisation pass over the D inputs is needed.
    alignas(64) T cr[4][UU], st[4][UU];
    for (int b = 0; b < 4; b++) for (int k = 0; k < UU; k++) { cr[b][k] = cbs[b][k] * rs[b * UU + k]; st[b][k] = cbs[b][k] * nq[b * UU + k]; }
    {
      const VU r0 = Q::load(cr[0]), r1 = Q::load(cr[1]), r2 = Q::load(cr[2]), r3 = Q::load(cr[3]);
      const VU s0 = Q::load(st[0]), s1 = Q::load(st[1]), s2 = Q::load(st[2]), s3 = Q::load(st[3]);
      for (int i = 0; i < D; i++) {
        const T ui = u[i]; T* HGLN_RESTRICT Hi = &H[(size_t)i * 4 * UU]; T* HGLN_RESTRICT pi = &p[(size_t)i * UU];
        VU h0 = Q::load(Hi), h1 = Q::load(Hi + UU), h2 = Q::load(Hi + 2 * UU), h3 = Q::load(Hi + 3 * UU);
        const VU pv = Q::load(pi);
        const VU dp = ((r0 * h0 * ui + r1 * h1 * ui) + r2 * h2 * ui) + r3 * h3 * ui;   // reference order: dp += cb_b*Hunit_b*u
        const VU pu = pv * ui;
        Q::store(Hi, h0 + s0 * pu); Q::store(Hi + UU, h1 + s1 * pu); Q::store(Hi + 2 * UU, h2 + s2 * pu); Q::store(Hi + 3 * UU, h3 + s3 * pu);
        alignas(64) T np[UU];
        Q::store(np, pv + dp);
        for (int k = 0; k < UU; k++) pi[k] = clipv(np[k], (T)0, (T)3);
      }
    }
    const T* zn = w.zn.data(); const T* ppv = w.pp.data();
    bool far = false;
    for (int b = 0; b < 4; b++) for (int k = 0; k < UU; k++) {
      const int j = b * UU + k; const T s = st[b][k];
      const T q2 = q[j] + (T)2 * s * (nq[j] * zn[j]) + s * s * ppv[k];
      q[j] = q2; nq[j] = std::sqrt(q2); rs[j] = (T)1 / nq[j];
      far |= !(q2 > (T)1e-6 && q2 < (T)1e6);
    }
    if (far) renormalise();                                                          // keep |H_b| near 1 (rare)
  }

  // Fixed connection budget: keep round_half_even(0.6*n) inputs per unit (n<=4: dense); bias column always kept.
  // M only ever loses columns, so after the first call no unit is above budget and later calls are no-ops; the
  // first call therefore compacts W to the surviving columns and frees A.
  void structure_step() {
    if (pruned) return;
    std::vector<int> act; act.reserve(n); std::vector<T> keep(n, 0);
    std::vector<std::vector<int>> keepcols(U);
    int newcs = 0;
    for (int k = 0; k < U; k++) {
      const int nc = ncol[k]; const int32_t* ck = &cols[(size_t)k * cs];
      act.clear();
      for (int q = 0; q < nc; q++) if (ck[q] < n) act.push_back(ck[q]);       // active inputs (bias excluded), ascending
      std::vector<char> drop(n, 0);
      if ((int)act.size() > nt) {
        for (int q = 0; q < nc; q++) {
          int j = ck[q]; if (j >= n) continue;
          T s = 0; const T* Wq = &W[((size_t)k * cs + q) * 16];
          for (int c = 0; c < 16; c++) s += std::fabs(Wq[c]);
          keep[j] = (s / 16) * A[(size_t)k * N1 + j];
        }
        std::stable_sort(act.begin(), act.end(), [&](int x, int y) { return keep[x] < keep[y]; });
        int nd = (int)act.size() - nt;
        for (int i = 0; i < nd; i++) drop[act[i]] = 1;
      }
      for (int q = 0; q < nc; q++) if (ck[q] >= n || !drop[ck[q]]) keepcols[k].push_back(q);   // slots kept, ascending
      newcs = std::max(newcs, (int)keepcols[k].size());
    }
    detail::ABuf<T> W2; W2.assign((size_t)U * newcs * 16, 0);
    detail::ABuf<int32_t> c2; c2.assign((size_t)U * newcs, 0);
    for (int k = 0; k < U; k++) {
      int nn = (int)keepcols[k].size();
      for (int q2 = 0; q2 < nn; q2++) {
        int q = keepcols[k][q2];
        c2[(size_t)k * newcs + q2] = cols[(size_t)k * cs + q];
        std::memcpy(&W2[((size_t)k * newcs + q2) * 16], &W[((size_t)k * cs + q) * 16], 16 * sizeof(T));
      }
      ncol[k] = nn;
    }
    W = std::move(W2); cols = std::move(c2); cs = newcs;
    A.reset(); pruned = true;
  }
};

#ifndef HGLN_V5_WBITS
#define HGLN_V5_WBITS 16
#endif
#ifndef HGLN_V5_ABITS
#define HGLN_V5_ABITS 16
#endif
// The defaults are version 5.0 (research/alg/tuned_v5.json, docs/ALGORITHM.md section 8): 8 units, fan-out, cross-layer
// latch 0.1 with credit 0.5, gates on the global error, glob 0.95, top-6 corners - the same model for the float Net (CPU
// and GPU) and the integer NetQ. Params::v43() / v42() give the earlier versions exactly.
struct Params {
  double glr0 = 0.2, glr1 = 0.05;   // gate learning rate schedule start/end (8-input streams); use 0.02/0.02 for 64-input classification
  double tau = 1000;                // schedule time constant (samples)
  double glob = 0.95;               // share of each unit's error taken from the global mixture error (4.x: 0.75)
  double mu = 0.3;                  // rule step (normalised LMS)
  int warm = 512;                   // first prune
  bool competence = false;          // true: version 4.2 competence-weighted mixture; false (4.3): plain average of the units
  int units = 8;                    // units per layer U: 8, 12 or 16 (others throw std::invalid_argument in Net::init).
                                    // 4.x: 16; 12 / 8 units: about 0.7x / 0.45x the work and state (docs/ALGORITHM.md)
  bool fanout = true;               // true: layer 3 reads layer 1's outputs instead of layer 2's (layers 2 and 3 then
                                    // run side by side: 2 sequential rule stages per sample); false (4.x): the 1-2-3 chain
  double latch = 0.1;               // > 0: cross-layer latch with dead band theta = latch on layers 2 and 3 (each bit
                                    // holds the same unit's bit of the layer it reads from - layer 1 with fanout - when
                                    // |z| < theta; stateless in time); 0: plain gates (versions 4.x)
  double latch_credit = 0.5;        // > 0: route the latch's held part of the gate gradient, times this scale, to the
                                    // lower layer's bit (layers are then updated 3, 2, 1); 0: off (4.x)
  int gate_err = 2;                 // error that drives the gate steps: 0 the mixed error es (4.x), 1 the unit's own error,
                                    // 2 the global mixture error y - out (5.0). Rule steps always use es.
  int corners = 6;                  // top-K corners per unit, 1..16 (16 = all, 4.x): the rules use the K largest heatmap
                                    // values renormalised, the gates learn through those K corners
  int wbits = HGLN_V5_WBITS, abits = HGLN_V5_ABITS;   // integer net: effective bits of the state (W, H, p) and of the
                                    // activations, 12..16 (the float Net ignores them)
  Params(double g0 = 0.2, double g1 = 0.05) : glr0(g0), glr1(g1) {}
  static Params v5(double g0 = 0.2, double g1 = 0.05) { return Params(g0, g1); }   // the defaults
  // version 4.3: 16 units, the 1-2-3 chain, plain gates, glob 0.75, all 16 corners
  static Params v43(double g0 = 0.2, double g1 = 0.05) {
    Params p(g0, g1); p.glob = 0.75; p.units = 16; p.fanout = false; p.latch = 0; p.latch_credit = 0; p.gate_err = 0; p.corners = 16;
    return p;
  }
  static Params v42(double g0 = 0.2, double g1 = 0.05) { Params p = v43(g0, g1); p.competence = true; return p; }   // 4.3 + competence
};

// The network: 3 layers x U units (Params::units), mixture over all 3U units (plain average, or competence-weighted).
// Per sample, the gates of all 3 layers (they depend only on the input) run first, then the rule stages in order.
template <class T> struct Net {
  typedef T real_type;
  typedef Work<T> work_type;
  static Params default_params(double g0 = 0.2, double g1 = 0.05) { return Params(g0, g1); }
  static constexpr long kRenormEvery = 4096;                 // exact renormalisation of the tracked hyperplane norms
  int d = 0; Layer<T> L[3]; long t = 0; Params P;
  Work<T> ws;                                                // scratch for predict()/learn()
  void init(int d_, uint64_t seed = 1, const Params& p = Params()) {
    if (!valid_units(p.units)) throw std::invalid_argument("hgln: Params::units must be 8, 12 or 16");
    if (p.corners < 1 || p.corners > 16) throw std::invalid_argument("hgln: Params::corners must be 1..16");
    if (p.gate_err < 0 || p.gate_err > 2) throw std::invalid_argument("hgln: Params::gate_err must be 0, 1 or 2");
    d = d_; P = p; t = 0;
    const int U = P.units;
    L[0].init(d, d + 1, P.competence, U); L[1].init(U, d + 1, P.competence, U); L[2].init(U, d + 1, P.competence, U);
    for (int l = 0; l < 3; l++) { L[l].latch = (T)P.latch; L[l].credit = (T)P.latch_credit; L[l].gate_err = P.gate_err; L[l].corners = P.corners; }
    Rng rng(seed); for (auto& l : L) l.random_hyperplanes(rng);
    ws.init(*this);
  }
  T out() const { return ws.out; }
  int units() const { return L[0].U; }
  const T* outputs() const { return ws.o.data(); }           // the 3U unit predictions from the last predict()

  // Forward pass into an arbitrary Work. const: safe to call from many threads, each with its own Work.
  T infer(const T* x, Work<T>& w) const {
    T* HGLN_RESTRICT u = w.u.data();
    T nn = 1; for (int i = 0; i < d; i++) { u[i] = x[i]; nn += x[i] * x[i]; }
    u[d] = 1; nn = std::sqrt(nn); for (int i = 0; i <= d; i++) u[i] /= nn;
    T* xg = w.tmp.data();
    const int U = L[0].U, NU = 3 * U;
    // gates first (independent of the previous layer), then the rule stages, which form the sequential chain
    L[0].gates(u, w.lw[0]); L[1].gates(u, w.lw[1], &w.lw[0]); L[2].gates(u, w.lw[2], &w.lw[P.fanout ? 0 : 1]);
    L[0].rules(x, w.lw[0], xg); L[1].rules(w.lw[0].pred.data(), w.lw[1], xg);
    L[2].rules(w.lw[P.fanout ? 0 : 1].pred.data(), w.lw[2], xg);   // fanout: layers 2 and 3 both read layer 1
    for (int l = 0; l < 3; l++) for (int k = 0; k < U; k++) w.o[l * U + k] = w.lw[l].pred[k];
    T out = 0;
    if (P.competence) {                                      // competence-weighted mixture (version 4.2)
      T cs = 0, comp[3 * kMaxUnits];
      for (int l = 0; l < 3; l++) for (int k = 0; k < U; k++) { comp[l * U + k] = w.lw[l].comp[k]; cs += comp[l * U + k]; }
      for (int i = 0; i < NU; i++) { w.r[i] = comp[i] / cs; out += w.r[i] * w.o[i]; }
    } else {                                                 // plain average (version 4.3)
      const T r = (T)1 / NU;
      for (int i = 0; i < NU; i++) { w.r[i] = r; out += r * w.o[i]; }
    }
    w.out = out;
    return out;
  }
  T predict(const T* x) { return infer(x, ws); }
  void learn(T y) {                   // must follow predict() on the same sample
    alignas(64) T es[3 * kMaxUnits];
    T gl = (T)P.glr1 + ((T)P.glr0 - (T)P.glr1) * hgln_exp<T>(-(T)t / (T)P.tau);   // hgln_exp: the GPU kernel's bits too
    T eg = y - ws.out, glob = (T)P.glob;
    const int U = L[0].U, NU = 3 * U;
    if (P.competence) {                // es_i = own error share + responsibility share r_i / sum r^2 of the mixture error
      T sr2 = 0;
      for (int i = 0; i < NU; i++) sr2 += ws.r[i] * ws.r[i];
      for (int i = 0; i < NU; i++) es[i] = (1 - glob) * (y - ws.o[i]) + glob * eg * ws.r[i] / sr2;
    } else {                           // equal weights: r_i / sum r^2 = 1
      for (int i = 0; i < NU; i++) es[i] = (1 - glob) * (y - ws.o[i]) + glob * eg;
    }
    UpdateArgs<T> a; a.y = y; a.mu = (T)P.mu; a.glr = gl; a.eg = eg; a.u = ws.u.data();
    if (P.latch > 0 && P.latch_credit > 0) {                 // latch credit: layers 3, 2, then 1 (which receives it)
      alignas(64) T acc[3][4 * kMaxUnits];
      for (auto& r : acc) for (auto& v : r) v = 0;
      for (int l = 2; l >= 0; l--) {
        const int lo = l == 0 ? -1 : (P.fanout ? 0 : l - 1);
        a.es = es + U * l; a.acc_in = acc[l]; a.acc_out = lo >= 0 ? acc[lo] : nullptr; a.lo = lo >= 0 ? &ws.lw[lo] : nullptr;
        L[l].update(a, ws.lw[l]);
      }
    } else {
      for (int l = 0; l < 3; l++) { a.es = es + U * l; a.lo = l == 0 ? nullptr : &ws.lw[P.fanout && l == 2 ? 0 : l - 1]; L[l].update(a, ws.lw[l]); }
    }
    t++;
    if (t >= P.warm && t % 64 == 0) for (auto& l : L) l.structure_step();
    if (t % kRenormEvery == 0) for (auto& l : L) l.renormalise();
  }
  T step(const T* x, T y) { T p = predict(x); learn(y); return p; }   // test-then-train on one sample
  bool pruned() const { return L[0].pruned && L[1].pruned && L[2].pruned; }
  size_t model_bytes() const { return L[0].model_bytes() + L[1].model_bytes() + L[2].model_bytes() + sizeof(*this); }
};

// Run many independent networks over one shared input stream, in parallel (one network per task). All nets must have
// the same d. X: N x d. Y: nets x N targets (row per network), or nullptr for predict only (model unchanged).
// P: nets x N outputs, or nullptr. Exactly equivalent to looping predict/learn sample by sample for each network.
template <class NetT, class T> void run_streams(std::vector<NetT>& nets, const T* X, const T* Y, T* P, size_t N, int threads = 0) {
  const int C = (int)nets.size();
  if (!C) return;
  const int d = nets[0].d;
  if (!Y) {                                                    // predict only: parallel over sample chunks (const path)
    const int chunk = 64, nch = (int)((N + chunk - 1) / chunk), nt = threads <= 0 ? hardware_threads() : threads;
    std::vector<typename NetT::work_type> works(std::max(1, std::min(nt, nch)));
    for (auto& w : works) w.init(nets[0]);
    parallel_for(nch, (int)works.size(), [&](int ch, int wi) {
      size_t a = (size_t)ch * chunk, b = std::min(N, a + chunk);
      for (int k = 0; k < C; k++)
        for (size_t t = a; t < b; t++) { T v = nets[k].infer(X + t * d, works[wi]); if (P) P[(size_t)k * N + t] = v; }
    });
    return;
  }
  parallel_for(C, threads, [&](int k, int) {
    NetT& g = nets[k]; const T* yk = Y + (size_t)k * N;
    for (size_t t = 0; t < N; t++) { T v = g.predict(X + t * d); g.learn(yk[t]); if (P) P[(size_t)k * N + t] = v; }
  });
}

// One-vs-rest classifier: one network per class on +1/-1 targets (NetT: Net<T>, or NetQ<T> from hgln_q.hpp).
// Heads share nothing, so they run in parallel: train_batch()/predict_batch() for blocks of samples, or, after
// set_threads(n), train()/predict() spread the heads of each single sample over a persistent pool.
template <class T, class NetT = Net<T>> struct Classifier {
  int d = 0, C = 0; std::vector<NetT> nets; std::vector<T> score;
  std::shared_ptr<ThreadPool> pool;                          // optional, for per-sample parallelism
  void init(int d_, int classes, uint64_t seed = 1, const Params& p = NetT::default_params(0.02, 0.02)) {
    d = d_; C = classes; nets.assign(C, NetT()); score.assign(C, 0);
    for (int k = 0; k < C; k++) nets[k].init(d, seed * 1000003ULL + k, p);
  }
  // Use n threads (0 = all hardware threads, 1 = none) inside train() and predict(). Worth it from about d >= 32 or
  // C >= 8; for tiny networks the hand-off costs more than it saves.
  void set_threads(int n) { int nt = n <= 0 ? hardware_threads() : n; pool = nt > 1 ? std::make_shared<ThreadPool>(nt) : nullptr; }
  void train(const T* x, int label) {
    if (pool && C > 1) {
      pool->run(C, [&](int k, int) { nets[k].predict(x); nets[k].learn(k == label ? (T)1 : (T)-1); });
      return;
    }
    for (int k = 0; k < C; k++) nets[k].predict(x);
    for (int k = 0; k < C; k++) nets[k].learn(k == label ? (T)1 : (T)-1);
  }
  int predict(const T* x) {                                  // prediction only: does not change the model
    if (pool && C > 1) pool->run(C, [&](int k, int) { score[k] = nets[k].predict(x); });
    else for (int k = 0; k < C; k++) score[k] = nets[k].predict(x);
    int best = 0;
    for (int k = 1; k < C; k++) if (score[k] > score[best]) best = k;
    return best;
  }
  // Train on N samples in order (X: N x d, labels: N). Same result as N calls to train(); heads run in parallel.
  void train_batch(const T* X, const int* labels, size_t N, int threads = 0) {
    parallel_for(C, threads, [&](int k, int) {
      NetT& g = nets[k];
      for (size_t t = 0; t < N; t++) { g.predict(X + t * d); g.learn(labels[t] == k ? (T)1 : (T)-1); }
    });
  }
  // Predict N samples (X: N x d) into out (N labels); optional scores (N x C). Parallel over samples; model unchanged.
  void predict_batch(const T* X, size_t N, int* out, T* scores = nullptr, int threads = 0) const {
    if (!C) return;
    const int chunk = 32, nch = (int)((N + chunk - 1) / chunk), nt = threads <= 0 ? hardware_threads() : threads;
    std::vector<typename NetT::work_type> works(std::max(1, std::min(nt, nch)));
    for (auto& w : works) w.init(nets[0]);
    parallel_for(nch, (int)works.size(), [&](int ch, int wi) {
      size_t a = (size_t)ch * chunk, b = std::min(N, a + chunk);
      for (size_t t = a; t < b; t++) {
        int best = 0; T bs = 0;
        for (int k = 0; k < C; k++) {
          T s = nets[k].infer(X + t * d, works[wi]);
          if (scores) scores[t * C + k] = s;
          if (k == 0 || s > bs) { bs = s; best = k; }
        }
        out[t] = best;
      }
    });
  }
  size_t model_bytes() const { size_t s = 0; for (auto& g : nets) s += g.model_bytes(); return s; }
};

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
// GCC reports -Wpsabi where it emits the out-of-line instantiation, usually the end of the including file, so the
// push/pop region above does not cover it; keep only this one warning off for the rest of the file.
#pragma GCC diagnostic ignored "-Wpsabi"
#endif
}  // namespace hgln
