// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_gpu.cpp - hgln::gpu::Bank (include/hgln_gpu.hpp): the HGLN kernel run through FORGE's gpu::cuda_hip runtime.
// The runtime dlopens the driver; its caller-module API (third_party/forge-patches/0001-*) loads HGLN's embedded
// PTX / code object and launches hgln_run on the runtime's stream. State stays on the device in the kernel's dense
// layout (cuda/hgln_dims.h); upload/download convert through the Layer accessors / import_dense.
// Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#include "hgln_gpu.hpp"
#include "hgln_dims.h"
#include "hgln_gpu_images.gen.hpp"   // generated at build time (cmake/hgln_gpu.cmake): hgln::gpu::images::table
#include <forge/gpu/cuda.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>

static_assert(sizeof(long) == 8, "the kernel's HgDims and tcount use 64-bit long (LP64)");

namespace hgln::gpu {
namespace fg = forge::gpu::cuda_hip;

static fg::api to_forge(Api a) { return a == Api::cuda ? fg::api::cuda : fg::api::hip; }

int device_count(Api api) {
  try { return fg::runtime::device_count(to_forge(api)); } catch (...) { return 0; }
}

template <class T> struct Bank<T>::Impl {
  std::unique_ptr<fg::runtime> rt;
  Api api = Api::cuda;
  Options opt;
  void* mod = nullptr; void* fn = nullptr;
  std::string image;
  std::size_t max_smem = 0;
  // networks on the device
  int nets = 0, d = 0; Params prm; HgDims g{};
  bool use_smem = false;
  std::vector<long> t;                                   // host copy of the sample counts
  std::uint64_t H = 0, Q = 0, p = 0, W = 0, M = 0, A = 0, m = 0, S = 0, tc = 0;
  std::uint64_t X = 0, Y = 0, P = 0; std::size_t capX = 0, capY = 0, capP = 0;   // per-call buffers (bytes)

  void free_state() {
    if (!rt) return;
    for (std::uint64_t* b : {&H, &Q, &p, &W, &M, &A, &m, &S, &tc}) if (*b) { rt->device_free(*b); *b = 0; }
    nets = 0; t.clear();
  }
  void free_io() {
    if (!rt) return;
    for (std::uint64_t* b : {&X, &Y, &P}) if (*b) { rt->device_free(*b); *b = 0; }
    capX = capY = capP = 0;
  }
  ~Impl() {
    try { free_state(); free_io(); if (rt && mod) rt->unload_module(mod); } catch (...) {}
  }
  void grow(std::uint64_t& b, std::size_t& cap, std::size_t bytes) {
    if (bytes <= cap) return;
    if (b) rt->device_free(b);
    b = 0; cap = 0; b = rt->device_alloc(bytes); cap = bytes;
  }
};

static bool fail(std::string* why, const std::string& s) { if (why) *why = s; return false; }

template <class T> Bank<T>::Bank() = default;
template <class T> Bank<T>::~Bank() = default;
template <class T> Bank<T>::Bank(Bank&&) noexcept = default;
template <class T> Bank<T>& Bank<T>::operator=(Bank&&) noexcept = default;

template <class T> bool Bank<T>::open(Api api, int ordinal, std::string* why, Options opt) {
  close();
  if (opt.threads < 1 || opt.threads > 1024) return fail(why, "threads per block must be 1..1024");
  constexpr int real = (int)sizeof(T);
  const char* rname = real == 8 ? "float64" : "float32";
  std::vector<const images::image*> cand;
  const char* force = std::getenv("HGLN_GPU_IMAGE");
  for (std::size_t i = 0; i < images::count; i++) {
    const auto& im = images::table[i];
    if (im.api != (api == Api::cuda ? 0 : 1) || im.real != real) continue;
    if (force && *force && !std::strstr(im.arch, force)) continue;
    cand.push_back(&im);
  }
  if (cand.empty())
    return fail(why, std::string(api_name(api)) + ": this build has no HGLN " + rname + " image" +
                         (force && *force ? std::string(" matching HGLN_GPU_IMAGE=") + force : std::string(" (clang lacked the GPU backend or lld at build time)")));
  auto im = std::make_unique<Impl>();
  im->api = api; im->opt = opt;
  try {
    std::string e;
    im->rt = fg::runtime::open(to_forge(api), &e, ordinal);
    if (!im->rt) return fail(why, e);
    std::string tried;
    for (auto* c : cand) {
      std::string me;
      void* mod = im->rt->load_module(std::string_view(reinterpret_cast<const char*>(c->data), c->size), &me);
      if (!mod) { tried += std::string(tried.empty() ? "" : "; ") + c->arch + ": " + me; continue; }
      void* fn = im->rt->module_function(mod, "hgln_run");
      void* tag = im->rt->module_function(mod, real == 8 ? "hgln_real_f64" : "hgln_real_f32");
      if (!fn || !tag) { im->rt->unload_module(mod); tried += std::string(tried.empty() ? "" : "; ") + c->arch + ": no hgln_run for " + rname; continue; }
      im->mod = mod; im->fn = fn; im->image = c->arch; break;
    }
    if (!im->mod) return fail(why, std::string(api_name(api)) + ": the device accepted no HGLN " + rname + " image (" + tried + ")");
    im->max_smem = im->rt->max_dynamic_smem();
  } catch (const std::exception& ex) {
    return fail(why, ex.what());
  }
  p_ = std::move(im);
  return true;
}

template <class T> bool Bank<T>::is_open() const { return p_ != nullptr; }
template <class T> void Bank<T>::close() { p_.reset(); }
template <class T> std::string Bank<T>::device_name() const { return p_ ? p_->rt->device_name() : std::string(); }
template <class T> std::string Bank<T>::image() const { return p_ ? p_->image : std::string(); }
template <class T> std::size_t Bank<T>::max_shared_bytes() const { return p_ ? p_->max_smem : 0; }
template <class T> std::size_t Bank<T>::size() const { return p_ ? (std::size_t)p_->nets : 0; }
template <class T> int Bank<T>::d() const { return p_ ? p_->d : 0; }
template <class T> bool Bank<T>::shared_scratch() const { return p_ && p_->use_smem; }
template <class T> std::size_t Bank<T>::scratch_bytes() const { return p_ && p_->nets ? (std::size_t)p_->g.scs * sizeof(T) : 0; }
template <class T> long Bank<T>::samples_seen(std::size_t n) const { return p_ && n < p_->t.size() ? p_->t[n] : 0; }

static bool same_params(const Params& a, const Params& b) {
  return a.glr0 == b.glr0 && a.glr1 == b.glr1 && a.tau == b.tau && a.glob == b.glob && a.mu == b.mu && a.warm == b.warm &&
         a.competence == b.competence && a.units == b.units && a.fanout == b.fanout &&
         a.latch == b.latch && a.latch_credit == b.latch_credit && a.gate_err == b.gate_err;
}

template <class T> bool Bank<T>::upload(const std::vector<Net<T>>& nets, std::string* why) {
  if (!p_) return fail(why, "bank is not open");
  Impl& I = *p_;
  if (nets.empty()) return fail(why, "no networks");
  const int d = nets[0].d;
  if (nets[0].P.latch != 0 || nets[0].P.latch_credit != 0 || nets[0].P.gate_err != 0)
    return fail(why, "the GPU kernel does not support the version 5 latch / latch credit / gate_err options yet");
  if (d < 1 || d > HG_MAXN) return fail(why, "d must be 1.." + std::to_string(HG_MAXN) + " (got " + std::to_string(d) + ")");
  for (auto& n : nets) {
    if (n.d != d) return fail(why, "all networks must have the same d");
    if (!same_params(n.P, nets[0].P)) return fail(why, "all networks must have the same Params (the kernel takes them once)");
  }
  const int U = nets[0].P.units;
  if (!hg_valid_units(U)) return fail(why, "Params::units must be 8, 12 or 16 (got " + std::to_string(U) + ")");
  for (auto& n : nets)
    if (n.units() != U || n.L[0].has_competence != n.P.competence) return fail(why, "a network's layers do not match its Params (not initialised?)");
  const HgDims g = hg_dims(d, U);
  const std::size_t N = nets.size(), smem = (std::size_t)g.scs * sizeof(T);
  // dense layout (cuda/hgln_dims.h), exactly as hg_pack in cuda/hgln_host.h: raw (unnormalised) hyperplanes and their
  // squared norms, the exact state the library keeps, so the device continues bit for bit
  std::vector<T> H(N * g.Hs, 0), Q(N * g.Qs, 0), p(N * g.ps, 0), W(N * g.Ws, 0), M(N * g.Ms, 0), A(N * g.Ms, 0), m(N * g.ms, 0);
  std::vector<long> t(N, 0);
  for (std::size_t n = 0; n < N; n++) {
    t[n] = nets[n].t;
    for (int l = 0; l < 3; l++) {
      const auto& L = nets[n].L[l]; const int N1 = L.N1, D = g.D;
      L.get_hyperplanes_raw(&H[n * g.Hs + l * 4L * U * D], &Q[n * g.Qs + l * 4L * U]);
      for (int k = 0; k < U; k++) {
        for (int i = 0; i < D; i++) p[n * g.ps + (l * (long)U + k) * D + i] = L.gate_weight(k, i);
        for (int c = 0; c < 16; c++) {
          m[n * g.ms + (l * (long)U + k) * 16 + c] = L.competence(k, c);
          for (int j = 0; j < N1; j++) W[n * g.Ws + g.wofs[l] + (k * 16L + c) * N1 + j] = L.rule(k, c, j);
        }
        for (int j = 0; j < N1; j++) {
          M[n * g.Ms + g.mofs[l] + k * N1 + j] = L.active(k, j) ? 1 : 0;
          A[n * g.Ms + g.mofs[l] + k * N1 + j] = L.activity(k, j);
        }
      }
    }
  }
  try {
    I.free_state();
    bool sh = false;
    if (I.opt.scratch != Scratch::global) {
      sh = smem <= I.max_smem && I.rt->set_dynamic_smem(I.fn, smem);
      if (!sh && I.opt.scratch == Scratch::shared)
        return fail(why, "scratch needs " + std::to_string(smem) + " bytes of shared memory per block; the device allows " + std::to_string(I.max_smem));
    }
    auto put = [&](std::uint64_t& b, const void* src, std::size_t bytes) { b = I.rt->device_alloc(bytes); I.rt->upload(b, src, bytes); };
    put(I.H, H.data(), H.size() * sizeof(T)); put(I.Q, Q.data(), Q.size() * sizeof(T)); put(I.p, p.data(), p.size() * sizeof(T)); put(I.W, W.data(), W.size() * sizeof(T));
    put(I.M, M.data(), M.size() * sizeof(T)); put(I.A, A.data(), A.size() * sizeof(T)); put(I.m, m.data(), m.size() * sizeof(T));
    put(I.tc, t.data(), t.size() * sizeof(long));
    if (!sh) I.S = I.rt->device_alloc(N * (std::size_t)g.scs * sizeof(T));
    I.use_smem = sh; I.g = g; I.d = d; I.prm = nets[0].P; I.nets = (int)N; I.t = t;
  } catch (const std::exception& ex) {
    I.free_state();
    return fail(why, ex.what());
  }
  return true;
}

template <class T> bool Bank<T>::run(const T* X, const T* Y, T* P, int samples, std::string* why) {
  if (!p_) return fail(why, "bank is not open");
  Impl& I = *p_;
  if (!I.nets) return fail(why, "no networks uploaded");
  if (samples < 0) return fail(why, "negative sample count");
  if (samples == 0) return true;
  if (!X || !Y) return fail(why, "X and Y are required (every sample is predicted and then learned)");
  try {
    const std::size_t xb = (std::size_t)samples * I.d * sizeof(T), yb = (std::size_t)I.nets * samples * sizeof(T);
    I.grow(I.X, I.capX, xb); I.grow(I.Y, I.capY, yb); I.grow(I.P, I.capP, yb);
    I.rt->upload(I.X, X, xb); I.rt->upload(I.Y, Y, yb);
    HgDims g = I.g;
    std::uint64_t S = I.use_smem ? 0 : I.S;
    int Ts = samples, warm = I.prm.warm, competence = I.prm.competence ? 1 : 0, units = g.U, fanout = I.prm.fanout ? 1 : 0;
    T glr0 = (T)I.prm.glr0, glr1 = (T)I.prm.glr1, tau = (T)I.prm.tau, glob = (T)I.prm.glob, mu = (T)I.prm.mu;
    // kernelParams in hgln_run's parameter order (cuda/hgln_kernel.h)
    void* args[] = {&g, &I.H, &I.Q, &I.p, &I.W, &I.M, &I.A, &I.m, &S, &I.X, &I.Y, &I.P, &I.tc, &Ts, &glr0, &glr1, &tau, &glob, &mu, &warm, &competence, &units, &fanout};
    const unsigned grid[3] = {(unsigned)I.nets, 1, 1}, block[3] = {(unsigned)I.opt.threads, 1, 1};
    I.rt->launch_kernel(I.fn, grid, block, I.use_smem ? (unsigned)((std::size_t)g.scs * sizeof(T)) : 0u, args);
    if (P) I.rt->download(P, I.P, yb);
    else I.rt->wait();
    for (auto& t : I.t) t += samples;
  } catch (const std::exception& ex) {
    return fail(why, ex.what());
  }
  return true;
}

template <class T> bool Bank<T>::run(const std::vector<T>& X, const std::vector<T>& Y, std::vector<T>& P, int samples, std::string* why) {
  if (!p_ || !p_->nets) return run(nullptr, nullptr, nullptr, samples, why);
  if (X.size() < (std::size_t)samples * p_->d || Y.size() < (std::size_t)samples * p_->nets) return fail(why, "X or Y is too short");
  P.resize((std::size_t)samples * p_->nets);
  return run(X.data(), Y.data(), P.data(), samples, why);
}

template <class T> bool Bank<T>::download(std::vector<Net<T>>& nets, std::string* why) const {
  if (!p_) return fail(why, "bank is not open");
  const Impl& I = *p_;
  if (!I.nets) return fail(why, "no networks uploaded");
  const HgDims& g = I.g; const std::size_t N = (std::size_t)I.nets;
  std::vector<T> H(N * g.Hs), Q(N * g.Qs), p(N * g.ps), W(N * g.Ws), M(N * g.Ms), A(N * g.Ms), m(N * g.ms);
  std::vector<long> t(N);
  try {
    I.rt->download(H.data(), I.H, H.size() * sizeof(T)); I.rt->download(Q.data(), I.Q, Q.size() * sizeof(T));
    I.rt->download(p.data(), I.p, p.size() * sizeof(T));
    I.rt->download(W.data(), I.W, W.size() * sizeof(T)); I.rt->download(M.data(), I.M, M.size() * sizeof(T));
    I.rt->download(A.data(), I.A, A.size() * sizeof(T)); I.rt->download(m.data(), I.m, m.size() * sizeof(T));
    I.rt->download(t.data(), I.tc, t.size() * sizeof(long));
  } catch (const std::exception& ex) {
    return fail(why, ex.what());
  }
  const std::size_t old = nets.size();
  if (old != N) nets.resize(N);
  for (std::size_t n = old; n < N; n++) nets[n].init(I.d, 1, I.prm);
  const long tp = hg_prune_time(I.prm.warm);
  for (std::size_t n = 0; n < N; n++) {
    auto& net = nets[n];
    if (net.d != I.d || net.units() != g.U || net.L[0].has_competence != I.prm.competence) net.init(I.d, 1, I.prm);   // layer shapes / map
    net.P = I.prm; net.t = t[n];
    const bool pruned = t[n] >= tp;                            // the kernel prunes once, at t == tp
    for (int l = 0; l < 3; l++)
      net.L[l].import_dense(&H[n * g.Hs + l * 4L * g.U * g.D], &p[n * g.ps + (long)l * g.U * g.D], &W[n * g.Ws + g.wofs[l]],
                            &M[n * g.Ms + g.mofs[l]], &A[n * g.Ms + g.mofs[l]], &m[n * g.ms + l * 16L * g.U], pruned, &Q[n * g.Qs + l * 4L * g.U]);
  }
  return true;
}

template class Bank<double>;
template class Bank<float>;

} // namespace hgln::gpu
