// SPDX-License-Identifier: AGPL-3.0-or-later
// charlm_hgln.cpp - character-level language modelling benchmark with HGLN on the two standard Wikipedia corpora, with
// the standard metric (bits per character / per byte, BPC) and split: characters [0, 90M) train, [90M, 95M) validation,
// [95M, 100M) test.
//   text8:  the cleaned first 10^8 characters of English Wikipedia (27 symbols, a-z and space; mattmahoney.net/dc/text8)
//           -> 27 one-vs-rest heads; input = one-hot of the previous K characters (d = 27 K; K = 4 -> 108)
//   enwik8: the raw first 10^8 bytes of the English Wikipedia XML dump (the Hutter Prize file, MD5
//           a1fa5ffddb56f4953e226637dabbb36a) -> 256 one-vs-rest heads (one per byte value); input = the 8 bits of each
//           of the previous K bytes as +-1 (d = 8 K; K = 16 -> 128)
// Each head is an HGLN network (Params() = version 5.0 with the 64-input classification gate rate 0.02), target +1 for
// the symbol that follows and -1 otherwise, one pass over the training characters, test-then-train. Probabilities:
// softmax of the head scores divided by one temperature, fitted on the first 1M validation characters (golden-section
// search on its log); the validation and test BPC use that temperature.
//
//   charlm_hgln FILE text8|enwik8 gpu|cpu [train_chars=90000000] [K] [chunk] [threads=0]
//   gpu: the heads train on the GPU (hgln::gpu::Bank, one thread block per head) and are evaluated on the CPU;
//   cpu: they train on the CPU (hgln::run_streams, heads in parallel) - the same model.
// Progress goes to stderr, one JSON line with the results to stdout.
// Copyright (C) Odin Loch. GNU AGPL v3.0 or later (see LICENSE).
#include "hgln_gpu.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
using namespace hgln;

struct Task {
  bool text8 = true; int C = 27, K = 4, d = 108;
  // features of position t (predicting s[t]) from s[t-K .. t-1] (nothing before the start)
  void features(const std::vector<unsigned char>& s, long t, float* x) const {
    if (text8) {
      std::fill(x, x + d, 0.0f);
      for (int i = 0; i < K; i++) { const long p = t - 1 - i; if (p >= 0) x[i * C + s[p]] = 1.0f; }
    } else {
      for (int i = 0; i < K; i++) { const long p = t - 1 - i; const int b = p >= 0 ? s[p] : 0;
        for (int j = 0; j < 8; j++) x[i * 8 + j] = p >= 0 ? (((b >> j) & 1) ? 1.0f : -1.0f) : 0.0f; }
    }
  }
};

// Scores of positions [t0, t0 + n) for every head, in parallel over positions (const infer, one Work per head and
// thread). keep: store the scores (n x C) in out; otherwise only accumulate bits and hits with temperature tau.
static void evaluate(const Task& T, const std::vector<Net<float>>& nets, const std::vector<unsigned char>& s, long t0, long n, int threads,
                     bool keep, std::vector<float>& out, double tau, double* bits, long* hits) {
  const int C = T.C;
  if (keep) out.assign((size_t)n * C, 0.0f);
  std::atomic<long> next(0); const long blk = 2048;
  std::vector<double> tb(threads, 0); std::vector<long> th(threads, 0);
  auto work = [&](int id) {
    std::vector<Work<float>> w(C); for (int c = 0; c < C; c++) w[c].init(nets[c]);
    std::vector<float> x(T.d), v(C);
    for (long b; (b = next.fetch_add(blk)) < n;)
      for (long i = b; i < std::min(n, b + blk); i++) {
        T.features(s, t0 + i, x.data());
        for (int c = 0; c < C; c++) v[c] = nets[c].infer(x.data(), w[c]);
        if (keep) { std::copy(v.begin(), v.end(), out.begin() + (size_t)i * C); continue; }
        double mx = -1e30; int am = 0;
        for (int c = 0; c < C; c++) if (v[c] > mx) { mx = v[c]; am = c; }
        double z = 0; for (int c = 0; c < C; c++) z += std::exp((v[c] - mx) / tau);
        const int y = s[t0 + i];
        tb[id] += -((v[y] - mx) / tau - std::log(z)) / std::log(2.0); th[id] += am == y;
      }
  };
  std::vector<std::thread> ts; for (int i = 0; i < threads; i++) ts.emplace_back(work, i); for (auto& t : ts) t.join();
  if (bits) { *bits = 0; for (double v : tb) *bits += v; }
  if (hits) { *hits = 0; for (long v : th) *hits += v; }
}
static double bpc_of(const std::vector<float>& sc, const std::vector<unsigned char>& s, long t0, long n, int C, double tau) {
  double bits = 0;
  for (long i = 0; i < n; i++) {
    const float* v = &sc[(size_t)i * C]; double mx = -1e30;
    for (int c = 0; c < C; c++) mx = std::max(mx, (double)v[c]);
    double z = 0; for (int c = 0; c < C; c++) z += std::exp((v[c] - mx) / tau);
    bits += -((v[s[t0 + i]] - mx) / tau - std::log(z)) / std::log(2.0);
  }
  return bits / n;
}

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: charlm_hgln FILE text8|enwik8 gpu|cpu [train_chars] [K] [chunk] [threads]\n"); return 1; }
  const std::string path = argv[1], kind = argv[2], mode = argv[3];
  Task T; T.text8 = kind == "text8";
  if (!T.text8 && kind != "enwik8") { std::fprintf(stderr, "dataset must be text8 or enwik8\n"); return 1; }
  T.C = T.text8 ? 27 : 256;
  T.K = argc > 5 ? std::atoi(argv[5]) : (T.text8 ? 4 : 16);
  T.d = T.text8 ? 27 * T.K : 8 * T.K;
  const long Ntr = argc > 4 ? std::atol(argv[4]) : 90000000L;
  const long chunk = argc > 6 ? std::atol(argv[6]) : std::max(10000L, 50000000L / T.C);
  int threads = argc > 7 ? std::atoi(argv[7]) : 0; if (threads <= 0) threads = hardware_threads();
  const int C = T.C, d = T.d;
  if (d > 128 && mode == "gpu") { std::fprintf(stderr, "the GPU kernel takes d <= 128\n"); return 1; }
  // ---- data
  std::vector<unsigned char> s(100000000);
  { FILE* f = std::fopen(path.c_str(), "rb"); if (!f) { std::perror(path.c_str()); return 1; }
    const size_t n = std::fread(s.data(), 1, s.size(), f); std::fclose(f);
    if (n != s.size()) { std::fprintf(stderr, "expected 10^8 bytes, read %zu\n", n); return 1; } }
  if (T.text8) for (size_t i = 0; i < s.size(); i++) {
    const unsigned char b = s[i]; const int v = b == ' ' ? 0 : (b >= 'a' && b <= 'z' ? b - 'a' + 1 : -1);
    if (v < 0) { std::fprintf(stderr, "text8: unexpected byte %d at %zu\n", b, i); return 1; }
    s[i] = (unsigned char)v;
  }
  const long V0 = 90000000L, T0 = 95000000L, NV = 5000000L, NT = 5000000L, NFIT = 1000000L;
  // ---- model
  const Params P(0.02, 0.02);
  std::vector<Net<float>> nets(C);
  for (int c = 0; c < C; c++) nets[c].init(d, 1 + c, P);
  std::fprintf(stderr, "%s: train %ld characters, %d heads, K=%d (d=%d), %s, %d units, top-%d corners, chunk %ld\n", kind.c_str(), Ntr, C, T.K, d, mode.c_str(), P.units, P.corners, chunk);
  // ---- training (test-then-train)
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
  const auto t_start = now();
  long okw = 0, nw = 0;                                      // online accuracy over the last million training characters
  std::vector<float> X, Y, Pr;
  gpu::Bank<float> bank; std::string why;
  if (mode == "gpu") {
    if (!bank.open(gpu::Api::cuda, 0, &why)) { std::fprintf(stderr, "no GPU: %s\n", why.c_str()); return 77; }
    if (!bank.upload(nets, &why)) { std::fprintf(stderr, "upload: %s\n", why.c_str()); return 1; }
    std::fprintf(stderr, "GPU: %s, image %s, scratch %zu bytes per head in %s memory\n", bank.device_name().c_str(), bank.image().c_str(),
                 bank.scratch_bytes(), bank.shared_scratch() ? "shared" : "global");
  }
  for (long t0 = 0; t0 < Ntr; t0 += chunk) {
    const long n = std::min(chunk, Ntr - t0);
    X.resize((size_t)n * d); Y.assign((size_t)C * n, -1.0f);
    for (long i = 0; i < n; i++) { T.features(s, t0 + i, &X[(size_t)i * d]); Y[(size_t)s[t0 + i] * n + i] = 1.0f; }
    const auto a = now();
    Pr.assign((size_t)C * n, 0.0f);
    if (mode == "gpu") { if (!bank.run(X.data(), Y.data(), Pr.data(), (int)n, &why)) { std::fprintf(stderr, "run: %s\n", why.c_str()); return 1; } }
    else run_streams(nets, X.data(), Y.data(), Pr.data(), (size_t)n, threads);
    const double dt = sec(a, now());
    if (t0 + n > Ntr - 1000000L) for (long i = 0; i < n; i++) {
      int am = 0; for (int c = 1; c < C; c++) if (Pr[(size_t)c * n + i] > Pr[(size_t)am * n + i]) am = c;
      okw += am == s[t0 + i]; nw++;
    }
    std::fprintf(stderr, "  trained %ld / %ld, %.2f us per character (all %d heads), elapsed %.0f s\n", t0 + n, Ntr, 1e6 * dt / n, C, sec(t_start, now()));
  }
  const double t_train = sec(t_start, now());
  if (mode == "gpu" && !bank.download(nets, &why)) { std::fprintf(stderr, "download: %s\n", why.c_str()); return 1; }
  // ---- evaluation on the CPU
  const auto e0 = now();
  std::vector<float> sv;
  evaluate(T, nets, s, V0, NFIT, threads, true, sv, 1.0, nullptr, nullptr);
  auto vb = [&](double lt) { return bpc_of(sv, s, V0, NFIT, C, std::exp(lt)); };
  double lo = std::log(0.01), hi = std::log(10.0); const double gr = (std::sqrt(5.0) - 1) / 2;
  double x1 = hi - gr * (hi - lo), x2 = lo + gr * (hi - lo), f1 = vb(x1), f2 = vb(x2);
  for (int it = 0; it < 40; it++) { if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - gr * (hi - lo); f1 = vb(x1); } else { lo = x1; x1 = x2; f1 = f2; x2 = lo + gr * (hi - lo); f2 = vb(x2); } }
  const double tau = std::exp((lo + hi) / 2);
  std::vector<float> none; double vbits, tbits; long vhit, thit;
  evaluate(T, nets, s, V0, NV, threads, false, none, tau, &vbits, &vhit);
  evaluate(T, nets, s, T0, NT, threads, false, none, tau, &tbits, &thit);
  const double t_eval = sec(e0, now());
  size_t bytes = 0; for (auto& n : nets) bytes += n.model_bytes();
  std::printf("{\"dataset\":\"%s\",\"model\":\"HGLN v5.0, %d one-vs-rest heads, K=%d context (d=%d), float32\",\"mode\":\"%s\",\"train_chars\":%ld,"
              "\"train_seconds\":%.1f,\"train_us_per_char\":%.3f,\"train_chars_per_second\":%.0f,\"online_train_acc_last_1M\":%.4f,"
              "\"temperature\":%.4f,\"valid_bpc\":%.4f,\"valid_acc\":%.4f,\"test_bpc\":%.4f,\"test_acc\":%.4f,\"eval_seconds\":%.1f,\"model_bytes\":%zu}\n",
              kind.c_str(), C, T.K, d, mode.c_str(), Ntr, t_train, 1e6 * t_train / Ntr, Ntr / t_train, nw ? (double)okw / nw : 0.0,
              tau, vbits / NV, (double)vhit / NV, tbits / NT, (double)thit / NT, t_eval, bytes);
  return 0;
}
