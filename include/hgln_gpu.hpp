// SPDX-License-Identifier: AGPL-3.0-or-later
// hgln_gpu.hpp - run many HGLN networks on a GPU through FORGE's CUDA/HIP runtime (forge::gpu::cuda_hip). No nvcc,
// no CUDA SDK, no ROCm at build time: the kernel (cuda/hgln_kernel.h) is compiled by clang into PTX (sm_80, sm_70)
// and AMDGPU code objects (gfx90a, gfx942, gfx1100 by default) for float64 and float32, embedded in the library, and
// the driver (libcuda / libamdhip64) is dlopened when a Bank is opened.
//
//   hgln::gpu::Bank<double> bank;
//   std::string why;
//   if (!bank.open(hgln::gpu::Api::cuda, 0, &why)) { /* no GPU: why says so; use the CPU library */ }
//   bank.upload(nets);                         // std::vector<hgln::Net<double>>, all with the same d and Params
//   bank.run(X, Y, P, T);                      // X: T x d (shared input stream), Y and P: nets x T (row per network)
//   bank.run(X2, Y2, P2, T2);                  // state stays on the device between calls
//   bank.download(nets);                       // back into the CPU networks (sample counts included)
//
// run() is exactly the CPU loop `for each sample: P = net.predict(x); net.learn(y)` for every network, one thread
// block per network. The per-sample scratch lives in dynamic shared memory when it fits (CUDA opt-in above 48 KB)
// and in global memory otherwise. A Bank is not thread-safe; use one per thread (each opens its own runtime).
// Needs the hgln_gpu library (CMake target hgln_gpu, built with FORGE). Copyright (C) Odin Loch. GNU AGPL v3.0+.
#pragma once
#include "hgln.hpp"
#include "hgln_q.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace hgln::gpu {

enum class Api { cuda, hip };
inline const char* api_name(Api a) { return a == Api::cuda ? "cuda" : "hip"; }
enum class Scratch { automatic, shared, global };   // where the kernel's per-sample scratch lives
inline const char* scratch_name(Scratch s) { return s == Scratch::shared ? "shared" : s == Scratch::global ? "global" : "automatic"; }

struct Options {
  int threads = 256;                        // threads per block (one block per network); any value 1..1024 is correct
  Scratch scratch = Scratch::automatic;     // automatic: shared memory if it fits on the device, else global
};

int device_count(Api api);                  // 0 when the driver is missing or has no devices

template <class T> class Bank {
public:
  Bank();
  ~Bank();
  Bank(Bank&&) noexcept;
  Bank& operator=(Bank&&) noexcept;
  Bank(const Bank&) = delete;
  Bank& operator=(const Bank&) = delete;

  // Open device `ordinal` and load the HGLN image for T that the device accepts (CUDA: sm_80 PTX, then sm_70;
  // HIP: the first code object the device loads; $HGLN_GPU_IMAGE, e.g. "sm_70" or "gfx1100", forces one).
  // false + why: no driver, no device, no image accepted, or this build has no image for that API.
  bool open(Api api, int ordinal = 0, std::string* why = nullptr, Options opt = {});
  bool is_open() const;
  void close();
  std::string device_name() const;          // as the driver reports it
  std::string image() const;                // the image in use: "ptx sm_80", "ptx sm_70", "gfx90a", ...
  std::size_t max_shared_bytes() const;     // dynamic shared memory one block can get on this device

  // Copy networks to the device (replacing any there). All must have the same d (1..128) and the same Params
  // (Params::competence included: the kernel runs the 4.3 plain average or, with competence, the 4.2 mixture;
  // Params::units = 8, 12 or 16 units per layer; Params::fanout: layer 3 reads layer 1, as in Net::infer; and the
  // version 5.0 options Params::latch, latch_credit, gate_err and corners, which all run on the GPU too).
  // The exact state is copied (unnormalised hyperplanes and their squared norms), so results match the CPU library.
  bool upload(const std::vector<Net<T>>& nets, std::string* why = nullptr);
  // Stream T samples through every network: predict, then learn. X: T x d, Y: nets x T, P: nets x T or nullptr.
  bool run(const T* X, const T* Y, T* P, int samples, std::string* why = nullptr);
  bool run(const std::vector<T>& X, const std::vector<T>& Y, std::vector<T>& P, int samples, std::string* why = nullptr);
  // Copy the device state back. nets is resized to the bank's size if needed (new entries take the uploaded
  // networks' d and Params); existing entries are overwritten, sample counts included.
  bool download(std::vector<Net<T>>& nets, std::string* why = nullptr) const;

  std::size_t size() const;                 // networks on the device
  int d() const;
  bool shared_scratch() const;              // true: scratch in dynamic shared memory
  std::size_t scratch_bytes() const;        // per network
  long samples_seen(std::size_t net) const; // as of the last run (host copy; no device access)

  struct Impl;
private:
  std::unique_ptr<Impl> p_;
};

extern template class Bank<double>;
extern template class Bank<float>;

// The integer network (hgln::NetQ<R>, version 5.0) on a GPU: the same calls as Bank, bit-identical to the CPU NetQ on
// any device (the kernel, cuda/hgln_q_kernel.h, is integer arithmetic). R (float or double) is the caller's real
// type for X, Y and P. The scratch is always in global memory (Options::scratch is ignored).
//   hgln::gpu::BankQ<float> bank;  bank.open(hgln::gpu::Api::cuda);  bank.upload(qnets);  bank.run(X, Y, P, T);
template <class R> class BankQ {
public:
  BankQ();
  ~BankQ();
  BankQ(BankQ&&) noexcept;
  BankQ& operator=(BankQ&&) noexcept;
  BankQ(const BankQ&) = delete;
  BankQ& operator=(const BankQ&) = delete;
  bool open(Api api, int ordinal = 0, std::string* why = nullptr, Options opt = {});
  bool is_open() const;
  void close();
  std::string device_name() const;
  std::string image() const;
  // All networks must have the same d (1..128) and Params.
  bool upload(const std::vector<NetQ<R>>& nets, std::string* why = nullptr);
  bool run(const R* X, const R* Y, R* P, int samples, std::string* why = nullptr);
  bool run(const std::vector<R>& X, const std::vector<R>& Y, std::vector<R>& P, int samples, std::string* why = nullptr);
  bool download(std::vector<NetQ<R>>& nets, std::string* why = nullptr) const;
  std::size_t size() const;
  int d() const;
  std::size_t scratch_bytes() const;        // per network (global memory)
  long samples_seen(std::size_t net) const;

  struct Impl;
private:
  std::unique_ptr<Impl> p_;
};

extern template class BankQ<double>;
extern template class BankQ<float>;

} // namespace hgln::gpu
