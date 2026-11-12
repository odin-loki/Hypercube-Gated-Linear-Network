// SPDX-License-Identifier: AGPL-3.0-or-later
// eval.cpp - score an algorithm variant (hgln_var.hpp) on the spec's benchmarks, all jobs in parallel.
//   eval DATA_DIR name key=val ...   keys: K U layers uniform no_p gate_src proj norm_track density glob mu
// Prints one JSON line: synthetic prequential NMSE per window (mean of 10 seeds, and per seed), digits holdout % per
// pass (mean of 5 shuffles, and per shuffle), and per-network cost (multiply-adds per sample, state values) at d=8, 64.
#include "hgln_var.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
using namespace hv;
static std::vector<double> load(const std::string& f) { FILE* p = fopen(f.c_str(), "rb"); std::vector<double> v; if (!p) { perror(f.c_str()); exit(1); }
  double b[4096]; size_t n; while ((n = fread(b, 8, 4096, p)) > 0) v.insert(v.end(), b, b + n); fclose(p); return v; }
int main(int argc, char** argv) {
  std::string dir = argv[1], name = argv[2]; Var V; int seed_off = 0;
  for (int i = 3; i < argc; i++) { std::string a = argv[i]; auto e = a.find('='); std::string k = a.substr(0, e); double v = atof(a.c_str() + e + 1);
    if (k == "K") V.K = (int)v; else if (k == "U") V.U = (int)v; else if (k == "layers") V.layers = (int)v; else if (k == "uniform") V.uniform = v != 0;
    else if (k == "no_p") V.no_p = v != 0; else if (k == "gate_src") V.gate_src = (int)v; else if (k == "proj") V.proj = (int)v;
    else if (k == "share_gates") V.share_gates = v != 0; else if (k == "p_shared") V.p_shared = v != 0;
    else if (k == "norm_track") V.norm_track = v != 0; else if (k == "density") V.density = v; else if (k == "glob") V.glob = v; else if (k == "mu") V.mu = v;
    else if (k == "theta") V.theta = v; else if (k == "theta_mode") V.theta_mode = (int)v; else if (k == "theta_lr") V.theta_lr = v; else if (k == "theta0") V.theta0 = v; else if (k == "sparse_k") V.sparse_k = (int)v; else if (k == "qbits") V.qbits = (int)v; else if (k == "wbits") V.wbits = (int)v; else if (k == "abits") V.abits = (int)v; else if (k == "shiftnorm") V.shiftnorm = v != 0;
    else if (k == "hybrid") V.hybrid = (int)v; else if (k == "pipeline") V.pipeline = v != 0; else if (k == "gate_every") V.gate_every = (int)v; else if (k == "seed_off") seed_off = (int)v;
    else { fprintf(stderr, "unknown key %s\n", k.c_str()); return 1; } }
  const char* tasks[3] = {"A", "B", "C"}; const int W[4][2] = {{0, 512}, {1536, 2048}, {2048, 2560}, {3584, 4096}};
  double syn[3][10][4]; double dig[5][3]; double ts[3][5][2]; const char* tsn[3] = {"narma10", "mackey", "regime"};
  std::atomic<int> next(0); int njobs = 50;
  auto work = [&] { for (int j; (j = next.fetch_add(1)) < njobs;) {
    if (j < 5) {   // digits shuffle 10 + j (longest jobs first)
      auto D = load(dir + "/dig_" + std::to_string(10 + j) + ".bin"); int n = (int)D.size() / 65, ntr = 2 * n / 3, C = 10;
      Var v = V; v.glr0 = v.glr1 = 0.02; std::vector<Net> nets(C); for (int k = 0; k < C; k++) nets[k].init(64, (10 + j) * 100 + k + 1 + 7919 * seed_off, v);
      for (int ep = 0; ep < 3; ep++) {
        for (int t = 0; t < ntr; t++) { const double* r = &D[(size_t)t * 65]; for (int k = 0; k < C; k++) { nets[k].predict(r + 1); nets[k].learn(k == (int)r[0] ? 1.0 : -1.0); } }
        int ok = 0;
        for (int t = ntr; t < n; t++) { const double* r = &D[(size_t)t * 65]; int best = 0; double bs = -1e9;
          for (int k = 0; k < C; k++) { double s = nets[k].predict(r + 1); if (s > bs) { bs = s; best = k; } } ok += best == (int)r[0]; }
        dig[j][ep] = 100.0 * ok / (n - ntr);
      }
    } else if (j < 20) {   // time-structured streams: NMSE over samples 2000-4000 and 4000-6000
      int q = j - 5, ti = q / 5, sd = 300 + q % 5;
      auto Dt = load(dir + "/ts_" + tsn[ti] + "_" + std::to_string(sd) + ".bin"); int dd = (int)Dt[0]; int N = (int)(Dt.size() - 1) / (dd + 1);
      Net net; net.init(dd, sd + 7919 * seed_off, V); std::vector<double> err(N), ys(N);
      for (int t = 0; t < N; t++) { const double* r = &Dt[1 + (size_t)t * (dd + 1)]; double p = net.predict(r); err[t] = (r[dd] - p) * (r[dd] - p); ys[t] = r[dd]; net.learn(r[dd]); }
      for (int w = 0; w < 2; w++) { int a0 = 2000 + 2000 * w, b0 = a0 + 2000; double se = 0, sm = 0, s2 = 0;
        for (int t = a0; t < b0; t++) { se += err[t]; sm += ys[t]; s2 += ys[t] * ys[t]; }
        ts[ti][q % 5][w] = (se / 2000) / (s2 / 2000 - (sm / 2000) * (sm / 2000)); }
    } else {
      int q = j - 20, ti = q / 10, sd = 200 + q % 10;
      auto D = load(dir + "/syn_" + tasks[ti] + "_" + std::to_string(sd) + ".bin"); int N = (int)D.size() / 9;
      Net net; net.init(8, sd + 7919 * seed_off, V); std::vector<double> err(N), ys(N);
      for (int t = 0; t < N; t++) { const double* r = &D[(size_t)t * 9]; double p = net.predict(r); err[t] = (r[8] - p) * (r[8] - p); ys[t] = r[8]; net.learn(r[8]); }
      for (int w = 0; w < 4; w++) { double se = 0, s = 0, s2 = 0; int c = W[w][1] - W[w][0];
        for (int t = W[w][0]; t < W[w][1]; t++) { se += err[t]; s += ys[t]; s2 += ys[t] * ys[t]; }
        syn[ti][q % 10][w] = (se / c) / (s2 / c - (s / c) * (s / c)); }
    } } };
  std::vector<std::thread> th; int nt = std::thread::hardware_concurrency(); for (int i = 0; i < nt; i++) th.emplace_back(work); for (auto& t : th) t.join();
  Net a, b; a.init(8, 1, V); b.init(64, 1, V);
  printf("{\"name\":\"%s\",\"syn\":{", name.c_str());
  for (int ti = 0; ti < 3; ti++) { printf("%s\"%s\":{\"mean\":[", ti ? "," : "", tasks[ti]);
    for (int w = 0; w < 4; w++) { double m = 0; for (int s = 0; s < 10; s++) m += syn[ti][s][w]; printf("%s%.4f", w ? "," : "", m / 10); }
    printf("],\"seeds\":["); for (int s = 0; s < 10; s++) { printf("%s[", s ? "," : ""); for (int w = 0; w < 4; w++) printf("%s%.4f", w ? "," : "", syn[ti][s][w]); printf("]"); } printf("]}"); }
  printf("},\"digits\":{\"mean\":["); for (int ep = 0; ep < 3; ep++) { double m = 0; for (int j = 0; j < 5; j++) m += dig[j][ep]; printf("%s%.2f", ep ? "," : "", m / 5); }
  printf("],\"shuffles\":["); for (int j = 0; j < 5; j++) printf("%s[%.2f,%.2f,%.2f]", j ? "," : "", dig[j][0], dig[j][1], dig[j][2]);
  printf("]},\"ts\":{");
  for (int ti = 0; ti < 3; ti++) { printf("%s\"%s\":{\"mean\":[", ti ? "," : "", tsn[ti]);
    for (int w = 0; w < 2; w++) { double m = 0; for (int s = 0; s < 5; s++) m += ts[ti][s][w]; printf("%s%.4f", w ? "," : "", m / 5); }
    printf("],\"seeds\":["); for (int s = 0; s < 5; s++) printf("%s[%.4f,%.4f]", s ? "," : "", ts[ti][s][0], ts[ti][s][1]); printf("]}"); }
  printf("},\"cost\":{\"d8\":{\"macs\":%.0f,\"state\":%zu},\"d64\":{\"macs\":%.0f,\"state\":%zu}}}\n", a.macs(), a.state_values(), b.macs(), b.state_values());
}
