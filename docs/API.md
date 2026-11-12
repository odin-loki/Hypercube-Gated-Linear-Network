# HGLN C++ API (`include/hgln.hpp`)

Header-only C++17. It needs only the standard library, and `-pthread` for the threaded calls. Namespace `hgln`. `T` is `float` or `double`. Licence: AGPL-3.0-or-later.

## Networks

```cpp
hgln::Net<double> net;
net.init(d, seed);                  // d inputs; Params() default: gate rate 0.2 -> 0.05 (suits ~8-input streams)
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
| `const T* outputs()`, `T out()` | The 48 unit predictions and the mixture output from the last `predict` |
| `bool pruned()`, `long t` | Whether the prune has run; samples learned |
| `size_t model_bytes()` | Bytes held by the model |

`Params`: `glr0`, `glr1`, `tau` (gate-rate schedule), `glob` (global-error share, 0.75), `mu` (rule step, 0.3) and `warm` (prune time, 512). `competence` (default `false`, version 4.3: the mixture is a plain average of the units; `true`, or `Params::v42()`, gives the 4.2 competence-weighted mixture exactly). The constants are documented in `docs/TGN_impl_spec.md` section 2.

`Work<T>`: per-sample scratch. Make one with `Work<T> w; w.init(net);`. One `Work` can serve any number of networks with the same `d`.

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
| `Bank<T>::open(api, ordinal, &why, Options{threads, scratch})` | Open a device through FORGE's runtime and load the matching HGLN image |
| `upload(nets)` | Copy networks (same `d` and `Params`) to the device |
| `run(X, Y, P, samples)` | X: samples x d (shared stream); Y, P: nets x samples. Same as the CPU predict/learn loop per network; state stays on the device |
| `download(nets)` | State and sample counts back into CPU networks |
| `image()`, `device_name()`, `shared_scratch()`, `scratch_bytes()` | What was chosen |

Environment: `HGLN_GPU_IMAGE` (force an image), `HGLN_GPU_SCRATCH=shared|global`, `HGLN_GPU_THREADS`. A `Bank` is not thread-safe; use one per thread.
