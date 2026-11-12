# HGLN C++ API (`include/hgln.hpp`)

Header-only C++17. It needs only the standard library, and `-pthread` for the threaded calls. Namespace `hgln`. `T` is `float` or `double`. Licence: AGPL-3.0-or-later.

## Networks

```cpp
hgln::Net<double> net;
net.init(d, seed);                  // d inputs; Params(): version 5.0, gate rate 0.2 -> 0.05 (suits ~8-input streams)
net.init(d, seed, hgln::Params(0.02, 0.02));   // fixed gate rate 0.02 (suited 64-input classification)
double yhat = net.predict(x);       // x: d values, scaled to about [-1, 1]
net.learn(y);                       // exactly once after each predict(), same sample
double p = net.step(x, y);          // predict then learn
```

| Member | Meaning |
|---|---|
| `T predict(const T* x)` | Forward pass. Stores what `learn` needs in the network's own `Work` |
| `void learn(T y)` | One online update from the last `predict`. The one-off prune runs inside it at sample 512 |
| `T infer(const T* x, Work<T>& w) const` | Forward pass into a caller-owned `Work`. Does not touch the model, so it is safe from many threads at once |
| `const T* outputs()`, `T out()` | The 3 x `units` unit predictions and the mixture output from the last `predict` |
| `bool pruned()`, `long t` | Whether the prune has run; samples learned |
| `size_t model_bytes()` | Bytes held by the model |

`Params` defaults are **version 5.0** (`Params()` = `Params::v5()`, `docs/ALGORITHM.md` section 8); `Params::v43()` and `Params::v42()` give versions 4.3 and 4.2 exactly. The constructor `Params(glr0, glr1)` sets the gate-rate schedule and keeps the other defaults. Fields, with the v5 value and the 4.3 value:

| Field | v5 | 4.3 | Meaning |
|---|---|---|---|
| `glr0`, `glr1`, `tau` | 0.2, 0.05, 1000 | same | gate-rate schedule `glr1 + (glr0 - glr1) exp(-t / tau)`; use 0.02 / 0.02 for 64-input classification |
| `glob` | 0.95 | 0.75 | share of each unit's error taken from the global mixture error |
| `mu`, `warm` | 0.3, 512 | same | rule step (normalised LMS); time of the one-off prune |
| `units` | 8 | 16 | units per layer: 8, 12 or 16 (anything else throws `std::invalid_argument`) |
| `fanout` | `true` | `false` | layers 2 and 3 both read layer 1 (2 sequential rule stages per sample instead of 3) |
| `latch` | 0.1 | 0 | > 0: cross-layer latch with that dead band on layers 2 and 3 |
| `latch_credit` | 0.5 | 0 | > 0 with the latch: the held part of the gate gradient goes to the lower layer |
| `gate_err` | 2 | 0 | error driving the gate steps: 0 the mixed error, 1 the unit's own error, 2 the global error |
| `corners` | 6 | 16 | top-K corners per unit, 1 to 16: the rules use the K largest heatmap values renormalised, the gates learn through them |
| `competence` | `false` | `false` | `true` (or `Params::v42()`): the 4.2 competence-weighted mixture |
| `wbits`, `abits` | 16, 16 | | integer net only: bits of the state and of the activations, 12 to 16 |

All of them work the same in the float `Net`, the integer `NetQ` (except `competence`, float only), on the CPU and on the GPU. The constants of the original algorithm are documented in `docs/TGN_impl_spec.md` section 2.

`Work<T>`: per-sample scratch. Make one with `Work<T> w; w.init(net);`. One `Work` can serve any number of networks with the same `d`.

## Integer network (`include/hgln_q.hpp`, version 5.0)

```cpp
#include "hgln_q.hpp"
hgln::NetQ<float> q;                 // the caller's real type for x, y and predictions (float or double)
q.init(d, seed);                     // Params::v5() by default: 8 units, fan-out, latch, top-6 corners, int16
float yhat = q.predict(x);
q.learn(y);
```

`NetQ<R>` has the same API as `Net<T>`: `predict`, `learn`, `step`, `infer(x, WorkQ&) const`, `out()`, `pruned()`, `t`, `model_bytes()`, and it works with `run_streams` and as `Classifier<R, NetQ<R>>`. State and activations are int16 (`Params::wbits` / `abits`, 12 to 16 bits), about half the memory of a float32 `Net`. The forward pass is integer arithmetic (int16 multiply-add dot products, a sigmoid table); learning is mixed: the per-unit control scalars stay integer (with three double-precision divisions), and the per-element updates of the gate weights, hyperplanes and rules are float32 on the int16 state, rounded back to the grid to nearest even. Everything is specified in `include/hgln_qspec.hpp` with a fixed operation order and no fused multiply-adds, so results are bit-identical on every CPU target (scalar, SSE2, AVX2, AVX-512 with or without VNNI, the plain-C++ fallback), on the GPU (`BankQ`) and to the research reference. Build with `-ffp-contract=off` (the Makefile and CMake builds do). It needs `Params::competence = false`; other out-of-range settings throw `std::invalid_argument`. `q.L[l]` has the inspection accessors `hyperplane`, `gate_weight`, `rule`, `active` and `activity`, returning the stored integers; `q.output(i)` is unit `i`'s last prediction.

## Classifier (one-vs-rest)

```cpp
hgln::Classifier<float> clf; clf.init(64, 10, /*seed=*/1);    // gate rate 0.02 by default
clf.train(x, label);                         // one sample, all heads
int c = clf.predict(x);                      // scores in clf.score
clf.set_threads(4);                          // optional: spread the heads of each train()/predict() over 4 threads
clf.train_batch(X, labels, N, threads);      // N samples in order; same result as N train() calls
clf.predict_batch(X, N, out, scores, threads);   // N labels (scores: N x C, may be nullptr)
```

`threads = 0` means all hardware threads. Results never depend on the thread count, because each head still learns its samples in order and heads share nothing.

## Many networks on one stream

```cpp
std::vector<hgln::Net<float>> nets(K);   // same d for all
hgln::run_streams(nets, X, Y, P, N, threads);        // X: N x d; Y, P: K x N (row per net); learns
hgln::run_streams(nets, X, nullptr, P, N, threads);  // predict only, parallel over samples, model unchanged
```

This is the CPU counterpart of the CUDA kernel (one block per network).

## Threading helpers

- `hgln::parallel_for(n, threads, fn)` calls `fn(item, worker)` for `item` in `[0, n)`. Items are handed out dynamically and the caller works too. It starts threads per call, so use it for blocks of work.
- `hgln::ThreadPool pool(threads); pool.run(n, fn);` does the same with persistent workers, for per-sample calls. Not re-entrant.

## Inspecting a layer (`net.L[0..2]`)

The internal layout is built for SIMD and keeps only active columns. These accessors return the values in the reference layout:

| Accessor | Reference value |
|---|---|
| `hyperplane(k, b, i)`, `get_hyperplanes(dst)` | `H[k][b][i]`, unit rows (dense U x 4 x D) |
| `hyperplane_raw`, `hyperplane_sqnorm`, `get_hyperplanes_raw(dst, sq)`, `set_hyperplanes(src, sq)` | The stored, unnormalised rows and their squared lengths (version 4.3 norm tracking); exact state |
| `gate_weight(k, i)` | `p[k][i]` |
| `rule(k, c, j)` | `W[k][c][j]` (0 for a pruned column) |
| `active(k, j)` | `M[k][j] > 0` |
| `competence(k, c)` | `m[k][c]` (0 without `Params::competence`) |
| `activity(k, j)` | `A[k][j]` (0 after the prune, when `A` is freed) |

## Build flags

- `-O3 -march=native` (or a fixed `-march`) is needed for the fast path. Without it the code is correct but runs at about the original speed.
- `-DHGLN_NO_VECTOR_EXT` forces the plain C++ fallback, for testing.
- Build with `-ffp-contract=off` (the Makefile and CMake builds do) for results that match the reference, every SIMD table and the GPU kernel to the last bit. g++ fuses multiply-adds by default even with `-std=c++17`, and clang does too. With contraction the results still pass parity (about 1e-15).

## FORGE builds (CMake): SIMD dispatch and GPU

See [FORGE.md](FORGE.md) for the build. Both need the CMake build; the header-only library does not.

**SIMD** (`include/hgln_simd.hpp`, target `hgln_simd`, compile with `HGLN_WITH_FORGE=1`): `Layer::forward`/`update` run through one kernel table chosen at run time. Nothing changes in user code.

| Call | Meaning |
|---|---|
| `hgln::simd::active().target` | Table in use: `"scalar"`, `"avx2"`, `"avx512"`, `"neon"`, `"sve"`, `"rvv"` |
| `hgln::simd::targets()` | Null-terminated list of tables built into the binary |
| `hgln::simd::force(name)` / `table(name)` | Use a specific table / get it (false / nullptr if this CPU cannot run it) |
| `hgln::simd::install(nullptr)` | Return to the automatic choice |
| `FORGE_SIMD_TARGET=name` | Environment override, as for FORGE |

**GPU** (`include/hgln_gpu.hpp`, target `hgln_gpu`):

| Call | Meaning |
|---|---|
| `hgln::gpu::device_count(Api::cuda / Api::hip)` | 0 if there is no driver or no device |
| `Bank<T>::open(api, ordinal, &why, Options{threads, scratch, kernel})` | Open a device through FORGE's runtime and load the matching HGLN image. `threads` = 0 (default): automatic (256 for fewer than 128 networks, else 64). `kernel`: `automatic` (= `block`, one thread block per network) or `warp` (CUDA only: one warp per network; same bits, not faster on an RTX 3090, see OPTIMISATION.md section 6) |
| `upload(nets)` | Copy networks (same `d` and `Params`) to the device |
| `run(X, Y, P, samples)` | X: samples x d (shared stream); Y, P: nets x samples. Same as the CPU predict/learn loop per network; state stays on the device |
| `download(nets)` | State and sample counts back into CPU networks |
| `image()`, `device_name()`, `shared_scratch()`, `scratch_bytes()`, `warp_kernel()`, `nets_per_block()`, `threads_per_block()` | What was chosen |

Environment: `HGLN_GPU_IMAGE` (force an image), `HGLN_GPU_SCRATCH=shared|global`, `HGLN_GPU_THREADS`, `HGLN_GPU_KERNEL=block|warp` (read by the CLI, the tests and the character LM). A `Bank` is not thread-safe; use one per thread.

`Bank<T>` runs the float `Net<T>`; **`BankQ<R>`** has the same calls for the integer `NetQ<R>` (its scratch is always in global memory). Both reproduce the CPU library **bit for bit** on the GPU, float64, float32 and int16, predictions and state (checked on an RTX 3090): the float kernel does the library's IEEE operations in the same order with the same `exp` (`hgln_exp`, `hgln_gate`), and the integer kernel is integer arithmetic.
