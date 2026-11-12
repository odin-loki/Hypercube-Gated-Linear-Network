# HGLN: C++ and CUDA Implementation Spec

**Author:** Odin Loch
**Status:** Version 2. Companion to `TGN_spec.md` (algorithm version 4.1). The C++ library is built, verified, SIMD-tuned and multithreaded (see `OPTIMISATION.md`). The CUDA kernel compiles to PTX with clang and is verified on a CPU emulator. It has **not been run on a GPU**, and the nvcc host program has not been compiled. Licence: AGPL-3.0-or-later.

## 1. What is verified and what is not

| Item | Status |
|---|---|
| C++17 library, float64 and float32 | Built and verified against the NumPy reference (section 6) |
| SIMD tuning, pruned-column compaction, multithreading | Done (section 8, `OPTIMISATION.md`); verified by the same parity tests and by threaded-equals-serial tests |
| Speed measurement | Measured before and after on one 4-vCPU machine (section 7) |
| CUDA / HIP kernel | Runs through FORGE's CUDA/HIP runtime (`FORGE.md`) and matches the library exactly against a mock driver. Built as PTX (sm_70, sm_80) and AMD code objects with clang, no SDK. Not run on real GPU hardware |
| Run-time SIMD dispatch | Done with FORGE: scalar, AVX2 and AVX-512 tables, bit-identical |

The library reproduces the algorithm (differences of about 1e-15 in float64 and about 1e-6 in float32 on prediction values). It is still slower than a matched small MLP, but the gap is now 3x at d = 8 and 3 to 9x at d = 64 (section 7).

## 2. Model constants

| Name | Value |
|---|---|
| Hidden layers, units per layer | 3, 16 (48 units total); no readout layer |
| Corners per unit (hyperplanes) | 16 (4) |
| Gate sharpness `BETA` | 12 |
| Gate-gradient clip | +-0.05 |
| Gate clip on `a` | [1e-4, 1 - 1e-4] |
| Unit output clip | [-2, 2] |
| Rule step `mu` | 0.3 |
| Gate rate | `glr_t = glr1 + (glr0 - glr1) * exp(-t / tau)`; `tau` = 1000 |
| 8-input streams | `glr0` = 0.2, `glr1` = 0.05 |
| 64-input classification | `glr0` = `glr1` = 0.02 |
| Global-gradient share `glob` | 0.75 |
| Competence rate, initial competence | 0.02, 0.3 |
| Competence floor in `comp` | 0.001 |
| Gate-weight clip `p` | [0, 3] |
| Activity decay | 0.98 / 0.02 |
| Warmup, consolidation period | 512, 64 |
| Connection density | 0.6: keep `round_half_even(0.6 * n)` inputs per unit (n = 8 -> 5, 16 -> 10, 64 -> 38); layers with n <= 4 stay dense |

## 3. State

`d` is the input size and `D = d + 1` is the side-information size. Layer 1 has `n = d` inputs. Layers 2 and 3 have `n = 16` inputs (the previous layer's 16 outputs). `N1 = n + 1` includes the +1 bias column. All arrays are per layer with U = 16 units.

| Array | Shape | Initial value | Changes |
|---|---|---|---|
| `H` | U x 4 x D | Rows are random normal vectors, last component set to 0, then each row normalised to unit length | Every sample |
| `p` | U x D | 1 | Every sample |
| `W` | U x 16 x N1 | 0 | Every sample |
| `M` | U x N1 | 1 | Once, at sample 512 (prune); the bias column is never pruned |
| `A` | U x N1 | 0.001 | Every sample |
| `m` | U x 16 | 0.3 | Every sample |

Per-sample scratch: `inp` (N1), `u` copy (D), `pu` (U x D), `a` (U x 4), `h` (U x 16), `f` (U x 16), `xm` (U x N1), `pred` (U), `comp` (U); and global `o`, `r` (48 each) and `out`.

**State that the NumPy reference carries but is dead in version 4.1, and is omitted from the port:** gate codes `E`, `alive`, `use`, `G`, the recent-input and recent-error buffers, the balancing offsets and their moving averages. With the fixed density, the consolidation step only ever prunes: after the first prune at sample 512 each unit already holds exactly the target number of inputs, so later consolidation calls change nothing. The parity test in section 6 confirms that dropping this state changes no output.

## 4. Exact procedure

> This section is the version 4.2 procedure (`Params::v42()`). Version 4.3, the library default, makes two changes, both specified in `ALGORITHM.md` section 1. It tracks hyperplane norms in closed form, which is the same model with less work. It uses a plain average of the units with `es_i = (1 - glob)(y - o_i) + glob (y - out)` and no competence map.

**Predict(x).**
1. `u = [x; 1] / ||[x; 1]||` (length D).
2. Layer 1 takes `a = x`; layer 2 takes layer 1's 16 outputs; layer 3 takes layer 2's 16 outputs.
3. For each unit of a layer, given input `a` (n values):
   1. `inp = [a; 1]`, `pu = p * u` (element-wise).
   2. `z_b = H_b . pu` for b = 0..3; `a_b = clip(1 / (1 + exp(-12 * z_b)), 1e-4, 1 - 1e-4)`.
   3. Corner weights, for corner index c = 0..15 with bit b of c: `h_c = product over b of (a_b if bit b of c is 1 else 1 - a_b)`.
   4. `xm = inp * M` (element-wise); `f_c = W_c . xm`; `mix = sum_c h_c * f_c`; `pred = clip(mix, -2, 2)`.
   5. `comp = 1 / (sum_c h_c * m_c + 0.001)`.
4. `o` = the 48 predictions (layer 1, then 2, then 3); `r_i = comp_i / sum_j comp_j`; `out = sum_i r_i * o_i`.

**Learn(y)**, called after Predict on the same sample.
1. `gl = glr1 + (glr0 - glr1) * exp(-t / tau)`.
2. `e_g = y - out`; for each unit i: `es_i = (1 - glob) * (y - o_i) + glob * e_g * r_i / sum_j r_j^2`.
3. For each unit, using the stored forward values and `es` of that unit, in this order:
   1. Own error `e = y - pred`. Competence: `m_c += 0.02 * h_c * (e^2 - m_c)`.
   2. `hf_c = h_c * f_c`; `tot = sum_c hf_c`; `F1_b = sum over corners with bit b = 1 of hf_c`; `F0_b = tot - F1_b`.
   3. `g_b = 12 * (F1_b * (1 - a_b) - F0_b * a_b)`; `c_b = clip(gl * es * g_b, -0.05, 0.05)`.
   4. `dp = sum_b c_b * H_b * u` (element-wise, using the **old** `H`); `p = clip(p + dp, 0, 3)`.
   5. `H_b += c_b * pu` using the **old** `pu`; renormalise each `H_b` to unit length.
   6. `c2 = mu * es / (0.001 + (sum_c h_c^2) * (sum_j xm_j^2))`; `W_c += c2 * h_c * xm` for every corner c.
   7. `A = 0.98 * A + 0.02 * |inp|` (element-wise).
4. `t += 1`. If `t >= 512` and `t mod 64 = 0`, run the consolidation step on every layer.

**Consolidation (prune), per unit.** Let `act` be the input columns j < n with `M_j > 0`. If `|act| > nt` (target count from section 2), compute `keep_j = mean over corners of |W_c,j| * A_j` for the active inputs, sort ascending (stable, ties by lower index), and set `M_j = 0` for the `|act| - nt` lowest. The bias column is untouched. Pruned columns keep their old `W` values but receive no updates and contribute nothing, because `xm` is masked.

**Classification.** One independent network per class, trained on +1 for its class and -1 otherwise; predict the class with the highest `out`. The networks share nothing, so they parallelise trivially.

## 5. Numerical notes

- The orders of summation differ between implementations, so expect differences around 1e-15 (float64) and 1e-6 (float32) in outputs. They do not grow over 2,000 samples in the tests here.
- `exp(-12 * z)` can overflow to infinity for very negative `z`; the result 1 / (1 + inf) = 0 is then clipped to 1e-4, so no special handling is needed on CPU. On GPU use a saturating sigmoid.
- Prune tie-breaking only matters for exactly equal scores, which did not occur in the tests.
- The rounding of `0.6 * n` must be round-half-even (as in `nearbyint`). The values 4.8 and 9.6 are not ties, so any rounding mode gives 5 and 10 here; for other n check the tie case.

## 6. Verification

**Parity protocol.** A harness writes a binary file: `int32 d`, `int32 N`, then the initial `H` of layers 1, 2, 3 as float64 (each U x 4 x D, row-major), then `X` (N x d, float64) and `y` (N, float64). The C++ program (`hgln parity in.bin out.bin [glr0 glr1]`) runs predict-then-learn per sample from those initial hyperplanes and writes N float64 predictions. The harness compares them with the NumPy reference `HGLN` run on the same data.

```python
net = HGLN(seed, d, glr0=glr0, glr1=glr1)          # NumPy reference from TGN_spec.md
with open("in.bin", "wb") as f:
    f.write(struct.pack("ii", d, N))
    for L in net.L: f.write(np.ascontiguousarray(L.H, dtype=np.float64).tobytes())
    f.write(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    f.write(np.ascontiguousarray(y, dtype=np.float64).tobytes())
P = np.zeros(N)
for t in range(N): P[t] = net.predict(X[t]); net.learn(y[t])
# run: ./hgln_d parity in.bin out.bin glr0 glr1   then   Q = np.fromfile("out.bin", dtype=np.float64)
```

**Results** (maximum absolute difference in predictions over a 2,000-sample parity run; the case also covers samples after the prune at sample 512):

| Test | Precision | Max abs difference |
|---|---|---|
| d = 8 | float64 | 2.6e-15 or below |
| d = 8 | float32 | 1.4e-06 or below |
| d = 64 | float64 | 2.6e-15 or below |
| d = 64 | float32 | 1.4e-06 or below |

These are the worst values across the four runs. The parity runs test arithmetic, not accuracy. The accuracy figures in `TGN_spec.md` came from the NumPy code.

## 7. Speed

**Optimised library** (version 2, this repository). Measured with `hgln_d|hgln_f bench d nets nsamples [threads]` on a 4-vCPU Xeon @ 2.1 GHz with AVX-512, `-O3 -march=native`. The original port was rebuilt in the same session for the first column. Figures are the middle of three runs, in microseconds per sample for all networks.

| Setting | Original | Optimised, 1 thread | Optimised, nets in parallel (4 threads) | MLP 3x16 |
|---|---|---|---|---|
| d = 8, 1 network, float64 | 14.3 (predict 7.4) | 6.0 (predict 2.65) | one network | 2.0 |
| d = 64, 10 networks, float64 | 492 (predict 251) | 246 (predict 130) | 52 (predict 20) | 26 |
| d = 64, 10 networks, float32 | 334 (predict 193) | 72.5 (predict 36) | 22.6 (predict 8.6) | 26 |

The rest of this section is the original measurement of the first port, kept for reference.

**First port** (version 1). Measured with `hgln bench d nets nsamples` (`-O3 -march=native`, single thread, shared 2-core machine). Timings vary between runs by about 20 percent, so two runs are shown. "Per sample" is one predict plus one learn for all networks. "MLP 3x16" is a matched baseline in the same file (d to 16 to 16 to 16 to 1, tanh, per-sample SGD); it allocates small vectors in its backward pass, so it is slightly pessimistic for the MLP.

| Setting | HGLN float64 (us) | HGLN float32 (us) | HGLN predict only, float64 (us) | MLP 3x16, float64 (us) |
|---|---|---|---|---|
| d = 8, 1 network | 11.1 and 14.4 | 11.6 and 12.4 | 5.4 to 6.7 | about 2.0 |
| d = 64, 10 networks (one-vs-rest) | 488 and 465 | 332 and 311 | 205 to 242 | 22 to 28 |

So the port is **about 6 to 7 times slower than a matched MLP at d = 8 and about 17 to 22 times slower at d = 64 with 10 heads**, single thread and unoptimised. Float32 helps at d = 64 (about 1.5 times) and not at d = 8. I did not measure the NumPy prototype against the port under identical conditions, so no speed-up over NumPy is claimed.

I have not counted operations per sample precisely; the time is dominated by 48 units each evaluating 16 rules over their active inputs, plus the gate update.

## 8. CPU optimisation roadmap (status in version 2)

Items 1, 2, 4 and 5 are done, measured and verified (`OPTIMISATION.md`). Item 1 was done as layouts that put units or corners in the vector lanes, rather than padding. Item 3 was not needed once the update was vectorised. Item 6 changes the algorithm and was not attempted. The original list:

1. Pad each row of `W`, `xm` and `H` to a multiple of the SIMD width and align them; the inner loops are short (9, 17 or 65 elements).
2. Compact the pruned input columns after sample 512 (40 percent of each unit's inputs are inactive), shrinking every `W` row and the forward and update loops.
3. Fuse the `W` update with the next sample's forward pass to reuse cache.
4. Run networks in parallel across threads (class heads, ensemble members, streams). This is the most reliable speed-up because the networks share nothing.
5. Use float32 (about 1.5 times faster at d = 64 in section 7; little change at d = 8).
6. Skip corners with negligible `h_c`. **This changes the algorithm** and must be re-validated for accuracy before use.

## 9. CUDA design (compiled to PTX and emulated; not run on a GPU)

The implemented kernel (`cuda/hgln_kernel.h`) uses one thread block per network, with all 48 units of that network inside the block and a block barrier between phases, rather than the per-unit mapping sketched below. That suits the target of many independent networks at once. Since version 2 the per-sample scratch can live in dynamic shared memory: 23 to 36 KB per block in float64 for d = 8 to 100. The p and hyperplane steps are fused, and dead work after the prune is skipped. `cuda/hgln_ptx.cu` builds the kernel to PTX without the CUDA SDK, using the same method as FORGE (`FORGE.md`).

The original design notes:

Mapping, as a starting point:

- One thread block per unit; 16 corners x (N1) elements of `W` handled by the block, one warp per corner group, with the dot products `f_c = W_c . xm` as warp reductions and the 4 hyperplane dot products by one warp.
- Layers depend on each other, and the mixture needs all 48 unit outputs, so a single network needs several synchronisation points per sample: layer 1, 2, 3, the mixture reduction, then the update. Use a persistent cooperative kernel with grid-wide synchronisation, or separate kernels.
- Keep `H`, `p`, `W`, `M`, `A`, `m` resident in global memory (W dominates: 16 x 16 x (n+1) values per layer) with no host round trips between samples.

Expected behaviour, which I have not measured: for a single stream at batch size 1, per-sample latency will be dominated by kernel launch and synchronisation costs, probably tens of microseconds, which is slower than the 11 microseconds of the CPU port at d = 8. The GPU is worth it when many independent networks run at once (class heads times streams times ensemble members). Plan the CUDA version around batching across networks, not around one network.

## 10. Acceptance criteria for any new implementation

1. Parity: float64 differences at most 1e-12 and float32 at most 1e-4 over the two parity cases in section 6, including the samples after the prune at sample 512.
2. Digits: the 5 shuffles in `TGN_spec.md`, one-vs-rest, gate rate 0.02, accuracy within about 1 point of the NumPy reference figures in that spec.
3. No heap allocation in the per-sample path.
4. A benchmark reported with the same settings as section 7, with the hardware stated.
5. For GPU: the same parity tests, plus a batched benchmark (for example 10 class heads across many streams).

## 11. Open item from the diversity check

Two findings from the extra experiments, both with limited seeds (10 per task for the ablation; the diversity check used the settings in `TGN_spec.md`):

- **Competence weighting adds almost nothing.** Replacing the competitive weights `r_u` with uniform weights (global-gradient share still 0.75) gives window-mean prequential NMSE (windows 1536-2048 / 3584-4096) of 0.125 / 0.082 on task A against 0.125 / 0.086 competitive, 0.129 / 0.166 on B against 0.132 / 0.172, and 0.151 / 0.212 on C against 0.149 / 0.213. All differences are within seed noise. The mixture behaves as a plain average of units trained with a shared global gradient. The competence map `m` and the `comp` computation could be dropped to simplify the port; this has been shown on these three tasks only, not on digits, so the reference keeps them for now.
- **Unit diversity falls as the global-gradient share rises.** With share 0 the mean effective number of units is 44.8 of 48; at 0.75 it is 45.2; at 1.0 it is 40.8. Layer-1 mean absolute pairwise correlation between unit outputs goes from 0.87 (share 0) to 0.78 (0.75) to 0.27 (1.0), the participation-ratio rank of the unit outputs rises from 1.25 to 1.52 to 4.8, and mean individual unit NMSE rises from 0.28 to 0.39 to 1.52 (layers 2 and 3 show the same pattern). At the default 0.75 the units stay individually useful; at 1.0 they become partial terms that only work in sum.

## 12. Reference C++ listing

Single file, no dependencies. Build: `g++ -O3 -march=native -std=c++17 hgln.cpp -o hgln_d` (float64) and add `-DUSE_FLOAT` for float32.

```cpp
// hgln.cpp - Hypercube Gated Linear Network (HGLN v4.1): single-thread C++17 reference port, no dependencies.
// Build:  g++ -O3 -march=native -std=c++17 hgln.cpp -o hgln_d            (float64, used for the parity test)
//         g++ -O3 -march=native -std=c++17 -DUSE_FLOAT hgln.cpp -o hgln_f (float32)
// Modes:  hgln parity in.bin out.bin [glr0 glr1] | hgln bench d nets nsamples
// Author: Odin Loch
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <string>
#ifdef USE_FLOAT
typedef float real;
#else
typedef double real;
#endif
static const real BETA = 12;
static inline real clipr(real v, real lo, real hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Layer {                       // 16 units; each unit = 4 hyperplanes -> 16-corner heatmap -> 16 linear rules
  int U, n, D, N1;                   // units, inputs, side-info size (d+1), inputs+bias
  std::vector<real> H, p, W, M, A, m;                 // learned / persistent state
  std::vector<real> inp, ucp, pu, a, h, f, xm, pred, comp;   // per-sample forward state
  void init(int U_, int n_, int D_) {
    U = U_; n = n_; D = D_; N1 = n + 1;
    H.assign(U * 4 * D, 0); p.assign(U * D, 1); W.assign(U * 16 * N1, 0); M.assign(U * N1, 1);
    A.assign(U * N1, (real)1e-3); m.assign(U * 16, (real)0.3);
    inp.assign(N1, 0); ucp.assign(D, 0); pu.assign(U * D, 0); a.assign(U * 4, 0); h.assign(U * 16, 0);
    f.assign(U * 16, 0); xm.assign(U * N1, 0); pred.assign(U, 0); comp.assign(U, 0);
  }
  void forward(const real* in, const real* u) {
    for (int j = 0; j < n; j++) inp[j] = in[j];
    inp[n] = 1;
    for (int i = 0; i < D; i++) ucp[i] = u[i];
    for (int k = 0; k < U; k++) {
      real* puk = &pu[k * D]; const real* pk = &p[k * D];
      for (int i = 0; i < D; i++) puk[i] = pk[i] * u[i];
      real* ak = &a[k * 4];
      for (int b = 0; b < 4; b++) {
        const real* Hb = &H[(k * 4 + b) * D]; real z = 0;
        for (int i = 0; i < D; i++) z += Hb[i] * puk[i];
        real s = 1 / (1 + std::exp(-BETA * z));
        ak[b] = clipr(s, (real)1e-4, (real)(1 - 1e-4));
      }
      real* hk = &h[k * 16];
      for (int c = 0; c < 16; c++) {
        real v = 1;
        for (int b = 0; b < 4; b++) v *= ((c >> b) & 1) ? ak[b] : (1 - ak[b]);
        hk[c] = v;
      }
      real* xk = &xm[k * N1]; const real* Mk = &M[k * N1];
      for (int j = 0; j < N1; j++) xk[j] = inp[j] * Mk[j];
      real mix = 0, cm = 0;
      for (int c = 0; c < 16; c++) {
        const real* Wc = &W[(k * 16 + c) * N1]; real s = 0;
        for (int j = 0; j < N1; j++) s += Wc[j] * xk[j];
        f[k * 16 + c] = s; mix += hk[c] * s; cm += hk[c] * m[k * 16 + c];
      }
      pred[k] = clipr(mix, (real)-2, (real)2);
      comp[k] = 1 / (cm + (real)1e-3);
    }
  }
  void update(real y, real mu, real glr, const real* es) {
    for (int k = 0; k < U; k++) {
      real e = y - pred[k], esk = es[k];
      real* hk = &h[k * 16]; real* mk = &m[k * 16];
      for (int c = 0; c < 16; c++) mk[c] += (real)0.02 * hk[c] * (e * e - mk[c]);   // competence map
      real tot = 0, F1[4] = {0, 0, 0, 0};
      for (int c = 0; c < 16; c++) {
        real hf = hk[c] * f[k * 16 + c]; tot += hf;
        for (int b = 0; b < 4; b++) if ((c >> b) & 1) F1[b] += hf;
      }
      real cb[4]; const real* ak = &a[k * 4];
      for (int b = 0; b < 4; b++) {                                                  // dm/dz_b
        real F0 = tot - F1[b], g = BETA * (F1[b] * (1 - ak[b]) - F0 * ak[b]);
        cb[b] = clipr(glr * esk * g, (real)-0.05, (real)0.05);
      }
      real* pk = &p[k * D]; const real* puk = &pu[k * D];
      for (int i = 0; i < D; i++) {                                                  // dp uses the OLD hyperplanes
        real dp = 0;
        for (int b = 0; b < 4; b++) dp += cb[b] * H[(k * 4 + b) * D + i] * ucp[i];
        pk[i] = clipr(pk[i] + dp, (real)0, (real)3);
      }
      for (int b = 0; b < 4; b++) {                                                  // hyperplane step + renormalise
        real* Hb = &H[(k * 4 + b) * D]; real nn = 0;
        for (int i = 0; i < D; i++) { Hb[i] += cb[b] * puk[i]; nn += Hb[i] * Hb[i]; }
        nn = std::sqrt(nn); for (int i = 0; i < D; i++) Hb[i] /= nn;
      }
      real h2 = 0, x2 = 0; const real* xk = &xm[k * N1];
      for (int c = 0; c < 16; c++) h2 += hk[c] * hk[c];
      for (int j = 0; j < N1; j++) x2 += xk[j] * xk[j];
      real c2 = (mu * esk) / ((real)1e-3 + h2 * x2);                 // normalised LMS
      for (int c = 0; c < 16; c++) {
        real* Wc = &W[(k * 16 + c) * N1]; real s = c2 * hk[c];
        for (int j = 0; j < N1; j++) Wc[j] += s * xk[j];
      }
      real* Ak = &A[k * N1];
      for (int j = 0; j < N1; j++) Ak[j] = (real)0.98 * Ak[j] + (real)0.02 * std::fabs(inp[j]);
    }
  }
  void structure_step() {            // fixed budget: keep round(0.6*n) inputs per unit (n<=4: dense); bias column always kept
    int nt = n <= 4 ? n : (int)std::nearbyint(0.6 * n);
    for (int k = 0; k < U; k++) {
      std::vector<int> act;
      for (int j = 0; j < n; j++) if (M[k * N1 + j] > 0) act.push_back(j);
      if ((int)act.size() > nt) {
        std::vector<real> keep(n, 0);
        for (int j = 0; j < n; j++) {
          real s = 0; for (int c = 0; c < 16; c++) s += std::fabs(W[(k * 16 + c) * N1 + j]);
          keep[j] = (s / 16) * A[k * N1 + j];
        }
        std::stable_sort(act.begin(), act.end(), [&](int x, int y) { return keep[x] < keep[y]; });
        int drop = (int)act.size() - nt;
        for (int i = 0; i < drop; i++) M[k * N1 + act[i]] = 0;
      }
    }
  }
};

struct HGLN {
  int d; Layer L[3]; real o[48], r[48], out, u[256]; long t; real glr0, glr1, tau, mu, glob; int warm;
  void init(int d_, real glr0_ = 0.2, real glr1_ = 0.05, real tau_ = 1000, real glob_ = 0.75) {
    d = d_; glr0 = glr0_; glr1 = glr1_; tau = tau_; mu = (real)0.3; glob = glob_; warm = 512; t = 0; out = 0;
    L[0].init(16, d, d + 1); L[1].init(16, 16, d + 1); L[2].init(16, 16, d + 1);
  }
  real predict(const real* x) {
    real nn = 1; for (int i = 0; i < d; i++) { u[i] = x[i]; nn += x[i] * x[i]; }
    u[d] = 1; nn = std::sqrt(nn); for (int i = 0; i <= d; i++) u[i] /= nn;
    L[0].forward(x, u); L[1].forward(L[0].pred.data(), u); L[2].forward(L[1].pred.data(), u);
    real cs = 0, comp[48];
    for (int l = 0; l < 3; l++) for (int k = 0; k < 16; k++) { o[l * 16 + k] = L[l].pred[k]; comp[l * 16 + k] = L[l].comp[k]; cs += comp[l * 16 + k]; }
    out = 0; for (int i = 0; i < 48; i++) { r[i] = comp[i] / cs; out += r[i] * o[i]; }
    return out;
  }
  void learn(real y) {
    real gl = glr1 + (glr0 - glr1) * std::exp(-(real)t / tau), sr2 = 0, es[48];
    for (int i = 0; i < 48; i++) sr2 += r[i] * r[i];
    real eg = y - out;
    for (int i = 0; i < 48; i++) es[i] = (1 - glob) * (y - o[i]) + glob * eg * r[i] / sr2;
    for (int l = 0; l < 3; l++) L[l].update(y, mu, gl, es + 16 * l);
    t++;
    if (t >= warm && t % 64 == 0) for (int l = 0; l < 3; l++) L[l].structure_step();
  }
};

struct Rng { uint64_t s; uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
  double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() { double a = uni() + 1e-300, b = uni(); return std::sqrt(-2 * std::log(a)) * std::cos(6.283185307179586 * b); } };

static void random_H(HGLN& g, Rng& rng) {
  for (int l = 0; l < 3; l++) { Layer& La = g.L[l];
    for (int k = 0; k < La.U * 4; k++) { double nn = 0; std::vector<double> v(La.D);
      for (int i = 0; i < La.D - 1; i++) { v[i] = rng.normal(); nn += v[i] * v[i]; } v[La.D - 1] = 0; nn = std::sqrt(nn);
      for (int i = 0; i < La.D; i++) La.H[k * La.D + i] = (real)(v[i] / nn); } }
}

struct MLP {                         // matched baseline: d -> 16 -> 16 -> 16 -> 1, tanh, per-sample SGD
  int d; std::vector<real> W[4], b[4], act[4]; real lr;
  void init(int d_, Rng& rng, real lr_ = 0.03f) { d = d_; lr = lr_; int sz[5] = {d, 16, 16, 16, 1};
    for (int i = 0; i < 4; i++) { W[i].resize(sz[i + 1] * sz[i]); b[i].assign(sz[i + 1], 0); act[i + 0].assign(sz[i + 1], 0);
      for (auto& w : W[i]) w = (real)(rng.normal() / std::sqrt((double)sz[i])); } }
  real predict(const real* x) { int sz[5] = {d, 16, 16, 16, 1}; const real* in = x; xin = in;
    for (int i = 0; i < 4; i++) { for (int o = 0; o < sz[i + 1]; o++) { real s = b[i][o]; for (int j = 0; j < sz[i]; j++) s += W[i][o * sz[i] + j] * in[j];
        act[i][o] = i < 3 ? std::tanh(s) : s; } in = act[i].data(); }
    return act[3][0]; }
  const real* xin;
  void learn(real y) { int sz[5] = {d, 16, 16, 16, 1}; std::vector<real> dl(1, act[3][0] - y);
    for (int i = 3; i >= 0; i--) { const real* in = i == 0 ? xin : act[i - 1].data(); std::vector<real> nd(sz[i], 0);
      for (int o = 0; o < sz[i + 1]; o++) { for (int j = 0; j < sz[i]; j++) { nd[j] += W[i][o * sz[i] + j] * dl[o]; W[i][o * sz[i] + j] -= lr * dl[o] * in[j]; } b[i][o] -= lr * dl[o]; }
      if (i > 0) for (int j = 0; j < sz[i]; j++) nd[j] *= (1 - act[i - 1][j] * act[i - 1][j]);
      dl = nd; } }
};

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "parity") {                       // in.bin: int32 d, int32 N, H0,H1,H2 (doubles), X (N*d), y (N)
    FILE* f = fopen(argv[2], "rb"); int32_t d, N; fread(&d, 4, 1, f); fread(&N, 4, 1, f);
    HGLN g; if (argc > 5) g.init(d, (real)atof(argv[4]), (real)atof(argv[5])); else g.init(d);
    for (int l = 0; l < 3; l++) { std::vector<double> buf(g.L[l].H.size()); fread(buf.data(), 8, buf.size(), f); for (size_t i = 0; i < buf.size(); i++) g.L[l].H[i] = (real)buf[i]; }
    std::vector<double> X((size_t)N * d), Y(N); fread(X.data(), 8, X.size(), f); fread(Y.data(), 8, Y.size(), f); fclose(f);
    std::vector<double> P(N); std::vector<real> x(d);
    for (int t = 0; t < N; t++) { for (int i = 0; i < d; i++) x[i] = (real)X[(size_t)t * d + i]; P[t] = g.predict(x.data()); g.learn((real)Y[t]); }
    f = fopen(argv[3], "wb"); fwrite(P.data(), 8, N, f); fclose(f); return 0;
  }
  if (mode == "bench") {                        // bench d nets nsamples
    int d = atoi(argv[2]), nets = atoi(argv[3]), N = atoi(argv[4]); Rng rng{88172645463325252ULL};
    std::vector<HGLN> G(nets); std::vector<MLP> M(nets);
    for (int k = 0; k < nets; k++) { G[k].init(d); random_H(G[k], rng); M[k].init(d, rng); }
    std::vector<real> x(d); auto gen = [&](real& y) { for (int i = 0; i < d; i++) x[i] = (real)(2 * rng.uni() - 1);
      y = (real)(std::sin(3 * x[0]) * std::cos(2 * x[1]) + 0.5 * x[2 % d] * x[3 % d]); };
    real y, sink = 0; int warm = 1500;
    for (int t = 0; t < warm; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += G[k].predict(x.data()); G[k].learn(y); } }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += G[k].predict(x.data()); G[k].learn(y); } }
    double th = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count() / N;
    t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) sink += G[k].predict(x.data()); }
    double tp = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count() / N;
    t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < N; t++) { gen(y); for (int k = 0; k < nets; k++) { sink += M[k].predict(x.data()); M[k].learn(y); } }
    double tm = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - t0).count() / N;
    printf("{\"d\":%d,\"nets\":%d,\"real\":\"%s\",\"hgln_us_per_sample\":%.2f,\"hgln_predict_only_us\":%.2f,\"mlp3x16_us_per_sample\":%.2f,\"sink\":%g}\n",
           d, nets, sizeof(real) == 4 ? "float32" : "float64", th, tp, tm, (double)sink);
    return 0;
  }
  fprintf(stderr, "usage: hgln parity in.bin out.bin [glr0 glr1] | hgln bench d nets nsamples\n"); return 1;
}
```
