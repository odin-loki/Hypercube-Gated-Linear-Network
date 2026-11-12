# HGLN: Hypercube Gated Linear Network

**Author:** Odin Loch (odin-loki). Version 4.3. Licence: [AGPL-3.0-or-later](LICENSE). Uses [FORGE](https://github.com/odin-loki/FORGE) for run-time SIMD dispatch and GPU execution ([docs/FORGE.md](docs/FORGE.md)).

HGLN is a stateless online learner: call `predict(x)`, then `learn(y)`, once per sample. It has three layers of 16 units. Each unit learns 4 hyperplanes that carve the input into 16 regions, and blends 16 linear rules by a soft heatmap over those regions. The output is a weighted mixture of all 48 units, and each unit learns partly from its share of the mixture's error. There is no backpropagation, no batching and no stored data.

**Version 4.3** changes the maths in two places, chosen after a measured study of 23 variants ([docs/ALGORITHM.md](docs/ALGORITHM.md)):

- **Exact norm tracking for the hyperplanes:** the same model, with two passes per hyperplane removed.
- **A plain average of the units instead of the competence-weighted mixture:** at least as accurate on the benchmarks, and it removes the competence map.

`Params::v42()` gives the previous version exactly. The study also shows the remaining structure (4 hyperplanes, 3 layers of 16, 60% density, per-unit gate weights) all earns its cost.

## Contents

| Path | What |
|---|---|
| `include/hgln.hpp` | Header-only C++17 library: `Net<T>` (regression), `Classifier<T>` (one-vs-rest), `run_streams`, thread helpers. API: [docs/API.md](docs/API.md) |
| `include/hgln_gate.h` | The gate nonlinearity, shared by the library and the CUDA kernel |
| `include/hgln_simd.hpp`, `src/simd/` | FORGE build: CPU kernels compiled per target (scalar, AVX2, AVX-512, NEON, SVE, RVV), chosen at run time |
| `include/hgln_gpu.hpp`, `src/gpu/` | FORGE build: `hgln::gpu::Bank`, many networks on a GPU through FORGE's CUDA/HIP runtime; `hgln_gpu` tool |
| `CMakeLists.txt`, `cmake/`, `third_party/forge-patches/` | CMake build that fetches and patches FORGE |
| `src/hgln_cli.cpp` | CLI: `parity`, `bench`, `classify` (built as `build/hgln_d` float64 and `build/hgln_f` float32) |
| `cuda/hgln_kernel.h` | GPU kernel: one block per network, many networks in parallel. Built for CUDA and HIP; **never run on a GPU** |
| `cuda/hgln_ptx.cu` | Header-free wrapper that builds the kernel to PTX / AMDGPU code with plain clang, no SDK |
| `cuda/hgln_cuda.cu` | nvcc host program (`selftest`, `bench`). **Never compiled** (no nvcc available) |
| `cuda/cuda_emu.cpp` | Runs the kernel source on the CPU and compares it with the C++ library |
| `python/hgln_ref.py` | NumPy reference implementation (the oracle) |
| `python/parity.py`, `python/digits_test.py` | NumPy-vs-C++ parity test and digits accuracy test |
| `tests/test_api.cpp` | C++ behavioural tests, including threaded-equals-serial checks |
| `tests/test_simd.cpp`, `tests/test_gpu.cpp`, `tests/mock_hgln.c` | Every SIMD table bit-identical; the full GPU path against a mock CUDA/HIP driver |
| `docs/TGN_spec.md`, `docs/TGN_impl_spec.md` | Algorithm spec and implementation spec |
| `docs/OPTIMISATION.md` | What was optimised, how it was checked, measurements |
| `docs/API.md`, `docs/FORGE.md` | C++ API reference; relation to FORGE |
| `docs/ALGORITHM.md`, `research/alg/` | Version 4.3 algorithm study: 23 variants on the spec's benchmarks |
| `docs/government_summary.md` | One-page summary for external readers |
| `research/` | Prototype scripts and stored experiment results |

## Build and test

**With FORGE (recommended).** You need clang 18 or g++ 14, CMake 3.25 or newer, Ninja, and Python 3 with numpy and scikit-learn. FORGE is fetched at a pinned commit and patched in the build tree.

    CXX=clang++-18 CC=clang-18 cmake -S . -B build/forge -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build/forge && ctest --test-dir build/forge     # 15 tests, about 90 s
    ./build/forge/hgln_cli info                                   # which SIMD table this CPU uses
    ./build/forge/hgln_gpu info                                   # GPUs found, or why not

This gives one portable binary that picks its CPU kernels at run time, plus the GPU library and tool. Pass `-DHGLN_FORGE_SOURCE_DIR=/path/to/FORGE` to use a local checkout.

**Header-only, no dependencies.** You need g++ or clang++ (C++20 for the emulator), and Python 3 with numpy and scikit-learn. `make ptx` also needs clang with the NVPTX backend.

    make            # builds into build/
    make test       # parity, emulator, PTX, digits (about 25 seconds)
    make bench      # speed benchmarks
    make ARCH=-march=x86-64-v3   # binaries for other AVX2 machines (default: -march=native)

## Use

```cpp
#include "hgln.hpp"
hgln::Net<double> net;
net.init(/*d=*/8, /*seed=*/1);          // 8-input stream: default gate rate 0.2 -> 0.05
double yhat = net.predict(x);           // x: pointer to 8 doubles
net.learn(y);                           // call once after each predict
```

Classification (64 inputs, 10 classes; gate rate fixed at 0.02, which suited 64 inputs):

```cpp
hgln::Classifier<float> clf; clf.init(64, 10, /*seed=*/1);
clf.train(x, label);                         // online, one sample
int c = clf.predict(x);                      // does not change the model
clf.train_batch(X, labels, N);               // N samples in order, heads in parallel; same result as N train() calls
clf.predict_batch(X, N, out);                // parallel over samples
clf.set_threads(4);                          // optional: parallel heads inside each single train()/predict()
```

Many independent networks on one stream, in parallel: `hgln::run_streams(nets, X, Y, P, N)` on the CPU, or `hgln::gpu::Bank` on a GPU (FORGE build, [docs/API.md](docs/API.md)). Concurrent read-only prediction from one network: `net.infer(x, work)`, with one `hgln::Work` per thread. Inputs should be scaled to roughly [-1, 1]. The gate rate depends on the task (spec section 5).

## Performance

Measured on a 4-vCPU Xeon with AVX-512, original and optimised builds in the same session (details and method in [docs/OPTIMISATION.md](docs/OPTIMISATION.md)):

| Microseconds per sample | Original | Header-only, `-march=native` | FORGE build (AVX-512 table), 1 thread | FORGE build, 4 threads | Matched MLP |
|---|---|---|---|---|---|
| d = 8, 1 network, float64 | 14.3 | 6.6 | 4.9 | (one network) | 2.0 |
| d = 64, 10 networks, float64 | 492 | 241 | 192 | 35 | 26 |
| d = 64, 10 networks, float32 | 334 | 75 | 78 | 20 | 26 |

Memory per network is about 33% lower after the prune at sample 512 (278 KB vs 417 KB at d = 64, float64). The FORGE build's portable binary is as fast as or faster than a native build. Its AVX2 table, for older CPUs, runs at about native speed in float64. The arithmetic is unchanged: float64 parity with NumPy is 1.8e-15 to 2.9e-15, the digits run gives the same 98.5%, and the CUDA kernel on the emulator matches the library exactly.

## What was tested (CPU only)

| Test | Result |
|---|---|
| C++ behavioural tests (determinism, learning, prune count, bounds, extreme inputs, XOR, threaded = serial, const inference, memory release at the prune) | pass |
| NumPy vs C++ parity, 2,000 samples, same start | float64 2.9e-15 (d = 8), 1.8e-15 (d = 64); float32 1.3e-6, 1.2e-6. Also with clang++, without `-march`, and with the non-vector fallback |
| CUDA kernel on the CPU emulator vs C++ | float64 exact (0 difference) for d = 8, 64, 100 and several thread counts, with scratch in global and in shared memory; float32 about 1e-6 |
| CUDA kernel compiled to PTX (clang 18, sm_70 and sm_80, float64 and float32) | compiles with `-Werror`, no unresolved calls |
| FORGE SIMD tables (scalar, AVX2, AVX-512) | bit-identical to each other and to the header-only build (predictions and full state) |
| GPU path through FORGE's CUDA/HIP runtime, against a mock driver that runs the real kernel on CPU threads | float64 exact (0 difference) vs the CPU library; CUDA and HIP; sm_80, sm_70, gfx90a, gfx1100; shared and global scratch; state resident over 5 calls; download and re-upload |
| Digits, C++ classifier, one shuffle, 3 passes | 98.5% holdout (599 samples), the same as before optimising. The C++ uses different random starting hyperplanes from the NumPy runs, so this is one more draw, not a replication of the 97.0% mean over 5 shuffles in the spec |

## What was not tested

- **A real GPU.** The kernel builds to PTX and AMD code objects and runs correctly through FORGE's runtime against a mock driver. No real driver has loaded it: ptxas, the JIT, ROCm, race behaviour on hardware and speed are all unknown. `hgln_cuda.cu` (nvcc) has never been compiled. For one stream at batch size 1, expect latency to be dominated by kernel launches; the design targets many networks at once.
- Real-world datasets other than digits; widths above 16; stronger baselines. The spec lists the selection effects behind its numbers.
- Inputs above 128 dimensions in the CUDA kernel (the prune scratch is fixed at 128; the C++ library has no such limit).
- Non-x86 CPUs. The NEON, SVE and RVV tables were only syntax-checked; the code was only run on x86-64.

## Notes

- `docs/TGN_impl_spec.md` embeds the first single-file C++ port (`research/hgln_v1_single_file.cpp`), kept as a historical record. The library is the same algorithm restructured for speed and passes the same parity tests.
- The competence weighting in the mixture added almost nothing in the ablation (spec section 7b). It is kept for fidelity with the reference.

## Licence

Copyright (C) 2026 Odin Loch. HGLN is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see [LICENSE](LICENSE) for details. Source files carry `SPDX-License-Identifier: AGPL-3.0-or-later`.
