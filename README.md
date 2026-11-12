# HGLN: Hypercube Gated Linear Network

**Author:** Odin Loch (odin-loki). Version 5.0. Licence: [AGPL-3.0-or-later](LICENSE). Uses [FORGE](https://github.com/odin-loki/FORGE) for run-time SIMD dispatch and GPU execution ([docs/FORGE.md](docs/FORGE.md)).

HGLN is a stateless online learner: call `predict(x)`, then `learn(y)`, once per sample. It has three layers of units, 8 per layer by default (`Params::units` = 8, 12 or 16). Each unit learns 4 hyperplanes that carve the input into 16 regions, and blends 16 linear rules by a soft heatmap over those regions. The output is a mixture of all 3U units (24 by default), and each unit learns partly from its share of the mixture's error. There is no backpropagation, no batching and no stored data.

**Version 4.3** (the previous version, `Params::v43()`) changed the maths in two places, chosen after a measured study of 23 variants ([docs/ALGORITHM.md](docs/ALGORITHM.md)):

- **Exact norm tracking for the hyperplanes:** the same model, with two passes per hyperplane removed.
- **A plain average of the units instead of the competence-weighted mixture:** at least as accurate on the benchmarks, and it removes the competence map.

`Params::v42()` gives version 4.2 exactly. The 4.3 study also measured two structural settings that trade accuracy for speed or parallelism against 4.3's chain of 3 layers of 16 units; both are now part of the 5.0 defaults:
- `Params::units = 12` or `8` (units per layer): about 0.6-0.7x or 0.35-0.5x the time and memory of 16 units, at a measured cost mostly in re-adaptation after a drift (8 is the default since 5.0).
- `Params::fanout`: 2 sequential rule stages per sample instead of the chain's 3 (the default since 5.0). The study also shows the remaining structure (4 hyperplanes, 3 layers, 60% density, per-unit gate weights) all earns its cost.

**Version 5.0** (`Params::v5()`, [docs/ALGORITHM.md](docs/ALGORITHM.md) section 8) takes the cheap structure, 8 units per layer with the fan-out (about 0.35x the work of 4.3), and wins the accuracy back with three learning-rule additions: a stateless cross-layer latch on the gates of layers 2 and 3, latch credit routed to layer 1, and gates that learn from the global error. Each unit also uses only its **top-6 corners** (the rules blend the 6 largest heatmap values, renormalised). Tuned on two seed sets and checked on a held-out third, it beats 4.3 on most benchmark windows and on digits (about +0.5 points); re-adaptation after the drift on stream B is the one place it is worse (+8% to +19% NMSE).

**Version 5.0 is the default** (`Params()`), for the float `Net` and the integer `NetQ`, on the CPU and the GPU. `Params::v43()` and `Params::v42()` give the earlier versions exactly.

**`hgln::NetQ`** (`include/hgln_q.hpp`) runs the same v5 model on an **int16 datapath**: half the memory of float32, an integer forward pass, and mixed learning steps (integer control scalars; float32 updates of the int16 state, rounded to nearest even). The operations are fixed, so the bits are the same on every CPU target and on the GPU. On digits it scores 97.33% (mean of 3 seed sets, 3 passes), against 97.05% for 4.3 and 97.54% for v5 in float. On one CPU core it is 1.3-2.0x slower than float32 running the same model ([docs/OPTIMISATION.md](docs/OPTIMISATION.md) section 5b).

**CPU and GPU give the same bits.** The GPU runs both networks: `hgln::gpu::Bank` (float) and `hgln::gpu::BankQ` (integer). On an NVIDIA RTX 3090 both reproduce the CPU library exactly, float64, float32 and int16, with predictions and the whole state identical. The float kernel does the library's IEEE operations in the same order and uses HGLN's own `exp` (no libm on either side). The integer kernel is integer arithmetic.

## Contents

| Path | What |
|---|---|
| `include/hgln.hpp` | Header-only C++17 library: `Net<T>` (regression), `Classifier<T>` (one-vs-rest), `run_streams`, thread helpers. API: [docs/API.md](docs/API.md) |
| `include/hgln_gate.h` | The gate nonlinearity, shared by the library and the CUDA kernel |
| `include/hgln_q.hpp`, `include/hgln_qspec.hpp` | Version 5.0 integer network `NetQ<R>` (same API as `Net`) and its exact integer specification |
| `include/hgln_simd.hpp`, `src/simd/` | FORGE build: CPU kernels compiled per target (scalar, AVX2, AVX-512, NEON, SVE, RVV), chosen at run time |
| `include/hgln_gpu.hpp`, `src/gpu/` | FORGE build: `hgln::gpu::Bank`, many networks on a GPU through FORGE's CUDA/HIP runtime; `hgln_gpu` tool |
| `CMakeLists.txt`, `cmake/`, `third_party/forge-patches/` | CMake build that fetches and patches FORGE |
| `src/hgln_cli.cpp` | CLI: `parity`, `bench`, `classify` (built as `build/hgln_d` float64, `build/hgln_f` float32 and `build/hgln_q` the integer net) |
| `cuda/hgln_kernel.h` | GPU kernel of the float `Net` (every option, v5 included): one block per network, many networks in parallel. Built for CUDA and HIP; run on an NVIDIA RTX 3090 |
| `cuda/hgln_q_kernel.h`, `cuda/hgln_q_dims.h`, `cuda/hgln_q_host.h`, `cuda/cuda_emu_q.cpp` | GPU kernel of the integer `NetQ` (`hgln::gpu::BankQ`), its layout, host helpers and CPU emulator test |
| `cuda/hgln_ptx.cu` | Header-free wrapper that builds the kernel to PTX / AMDGPU code with plain clang, no SDK |
| `cuda/hgln_cuda.cu` | nvcc host program (`selftest`, `bench`); compiled with nvcc 12.0 and run on the RTX 3090 |
| `cuda/cuda_emu.cpp` | Runs the kernel source on the CPU and compares it with the C++ library |
| `python/hgln_ref.py` | NumPy reference implementation (the oracle) |
| `python/parity.py`, `python/digits_test.py` | NumPy-vs-C++ parity test and digits accuracy test |
| `tests/test_api.cpp` | C++ behavioural tests, including threaded-equals-serial checks |
| `tests/test_q.cpp` | The integer net against the research reference of its specification, bit for bit |
| `tests/test_simd.cpp`, `tests/test_gpu.cpp`, `tests/mock_hgln.c` | Every SIMD table bit-identical; the full GPU path against a mock CUDA/HIP driver |
| `docs/TGN_spec.md`, `docs/TGN_impl_spec.md` | Algorithm spec and implementation spec |
| `docs/OPTIMISATION.md` | What was optimised, how it was checked, measurements |
| `docs/API.md`, `docs/FORGE.md` | C++ API reference; relation to FORGE |
| `docs/ALGORITHM.md`, `research/alg/` | Algorithm studies: version 4.3 (23 variants) and version 5.0 (latch, credit, gate error, tuning, integer datapath, rounding) |
| `docs/government_summary.md` | One-page summary for external readers |
| `research/` | Prototype scripts and stored experiment results |

## Build and test

**With FORGE (recommended).** You need clang 18 or g++ 14, CMake 3.25 or newer, Ninja, and Python 3 with numpy and scikit-learn. FORGE is fetched at a pinned commit and patched in the build tree.

    CXX=clang++-18 CC=clang-18 cmake -S . -B build/forge -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build/forge && ctest --test-dir build/forge     # 100 tests, several minutes (the GPU mock tests take most of it)
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
net.init(/*d=*/8, /*seed=*/1);          // version 5.0 defaults; 8-input stream: gate rate 0.2 -> 0.05
                                        // (hgln::Params::v43() as a third argument: version 4.3)
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

Version 4.3 settings (16 units; `Params::v43()`), measured on a 4-vCPU Xeon with AVX-512, original and optimised builds in the same session (details and method in [docs/OPTIMISATION.md](docs/OPTIMISATION.md)). The version 5.0 defaults do about 0.35x the work; the integer net's measurements are in section 5 there.

| Microseconds per sample | Original | Header-only, `-march=native` | FORGE build (AVX-512 table), 1 thread | FORGE build, 4 threads | Matched MLP |
|---|---|---|---|---|---|
| d = 8, 1 network, float64 | 14.3 | 6.6 | 4.9 | (one network) | 2.0 |
| d = 64, 10 networks, float64 | 492 | 241 | 192 | 35 | 26 |
| d = 64, 10 networks, float32 | 334 | 75 | 78 | 20 | 26 |

Memory per network is about 33% lower after the prune at sample 512 (278 KB vs 417 KB at d = 64, float64). The FORGE build's portable binary is as fast as or faster than a native build. Its AVX2 table, for older CPUs, runs at about native speed in float64. The arithmetic is unchanged: float64 parity with NumPy is 1.8e-15 to 2.9e-15, the digits run gives the same 98.5%, and the CUDA kernel on the emulator matches the library exactly.

## What was tested

| Test | Result |
|---|---|
| C++ behavioural tests (determinism, learning, prune count, bounds, extreme inputs, XOR, threaded = serial, const inference, memory release at the prune) | pass |
| NumPy vs C++ parity, 2,000 samples, same start | float64 2.9e-15 (d = 8), 1.8e-15 (d = 64); float32 1.3e-6, 1.2e-6. Also with clang++, without `-march`, and with the non-vector fallback |
| CUDA kernel on the CPU emulator vs C++ | float64 exact (0 difference) for d = 8, 64, 100 and several thread counts, with scratch in global and in shared memory; float32 exact (0 difference) too |
| CUDA kernel compiled to PTX (clang 18, sm_70 and sm_80, float64 and float32) | compiles with `-Werror`, no unresolved calls |
| FORGE SIMD tables (scalar, AVX2, AVX-512) | bit-identical to each other and to the header-only build (predictions and full state) |
| GPU path through FORGE's CUDA/HIP runtime, against a mock driver that runs the real kernel on CPU threads | float64 exact (0 difference) vs the CPU library; CUDA and HIP; sm_80, sm_70, gfx90a, gfx1100; shared and global scratch; state resident over 5 calls; download and re-upload |
| Digits, C++ classifier, one shuffle, 3 passes | 98.5% holdout (599 samples), the same as before optimising. The C++ uses different random starting hyperplanes from the NumPy runs, so this is one more draw, not a replication of the 97.0% mean over 5 shuffles in the spec |
| Integer net (`NetQ`) vs the research reference of its spec | bit-identical (predictions and full state) for d = 8 and 64, 8/12/16 units, 16 and 12 bits, v5 and 4.3-like settings, past the prune and 4,096 samples; on scalar, SSE2, AVX2, AVX-512 with and without VNNI, the plain-C++ fallback, g++ and clang. Digits end-to-end: 99.0% on the same shuffle |
| GPU kernels on the CPU emulator vs the C++ library | float64 and float32 exact (0 difference) for every option set: 4.3, 4.2, the v5 defaults, latch with and without credit, own- and global-error gates, top-6 / top-8 corners; 8/12/16 units; fan-out and chain; d = 8, 64, 100; across the prune and t = 4096. The integer kernel: bit-identical to `NetQ` (predictions and every stored integer), 16 and 12 bits, and back into a `NetQ` |
| **Real GPU: NVIDIA RTX 3090** (sm_86, CUDA driver under WSL2) through FORGE's runtime (`hgln_gpu selftest`, `Bank` and `BankQ`) and the nvcc host program (`hgln_cuda selftest`) | The driver JIT-loads the sm_80 PTX. GPU vs CPU library: **0 difference** in predictions and state, float64, float32 and the integer net; 4.3, v4.2 and v5, 8/12/16 units, fan-out and chain, d = 8, 64, 100, up to 4,500 samples |

## What was not tested

- **AMD GPUs and other NVIDIA generations.** Only one RTX 3090 has run the kernels. The AMD code objects have not been loaded by ROCm. GPU speed has not been studied: one block per network is slow for a single stream (tens of µs per sample, dominated by barriers and launches); the design targets many networks at once.
- Real-world datasets other than digits; widths above 16; stronger baselines. The spec lists the selection effects behind its numbers.
- Inputs above 128 dimensions in the CUDA kernel (the prune scratch is fixed at 128; the C++ library has no such limit).
- Non-x86 CPUs. The NEON, SVE and RVV tables were only syntax-checked; the code was only run on x86-64.

## Notes

- `docs/TGN_impl_spec.md` embeds the first single-file C++ port (`research/hgln_v1_single_file.cpp`), kept as a historical record. The library is the same algorithm restructured for speed and passes the same parity tests.
- The competence weighting in the mixture added almost nothing in the ablation (spec section 7b). It is off by default since 4.3 and kept as an option for fidelity with the reference (`Params::v42()` or `competence = true`).

## Licence

Copyright (C) 2026 Odin Loch. HGLN is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see [LICENSE](LICENSE) for details. Source files carry `SPDX-License-Identifier: AGPL-3.0-or-later`.
