# HGLN: Hypercube Gated Linear Network

**Author:** Odin Loch (odin-loki). Version 4.2.

A stateless online learner. Call `predict(x)`, then `learn(y)`, once per sample. Three layers of 16 units; each unit learns 4 hyperplanes
that carve the input into 16 regions and blends 16 linear rules by a soft heatmap over those regions. The output is a weighted mixture of all
48 units, and each unit learns partly from its share of the mixture's error. No backpropagation, no batches, no stored data.

## Contents

| Path | What |
|---|---|
| `include/hgln.hpp` | Header-only C++17 library: `Net<T>` (regression), `Classifier<T>` (one-vs-rest), `Params` |
| `src/hgln_cli.cpp` | `hgln_cli`-style tool: `parity`, `bench`, `classify` (built as `build/hgln_d` float64 and `build/hgln_f` float32) |
| `cuda/hgln_kernel.h` | CUDA kernel, one block per network, many networks in parallel. **Never run on a GPU** |
| `cuda/hgln_cuda.cu` | CUDA host program (`selftest`, `bench`). **Never compiled** (no CUDA compiler was available) |
| `cuda/cuda_emu.cpp` | Runs the kernel source on the CPU and compares it with the reference C++ |
| `python/hgln_ref.py` | NumPy reference implementation |
| `python/parity.py`, `python/digits_test.py` | NumPy-vs-C++ parity test and digits accuracy test |
| `tests/test_api.cpp` | C++ behavioural tests |
| `docs/TGN_spec.md`, `docs/TGN_impl_spec.md` | Algorithm spec and implementation spec |
| `docs/government_summary.md` | One-page summary for external readers |
| `research/` | Prototype scripts and stored experiment results |

## Build and test

Needs g++ (C++20 for the emulator), Python 3 with numpy and scikit-learn.

    make          # builds into build/
    make test     # runs everything below (about 10 seconds)

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
hgln::Classifier<double> clf; clf.init(64, 10, /*seed=*/1);
clf.train(x, label);                    // online
int c = clf.predict(x);                 // does not change the model
```

Inputs should be scaled to roughly [-1, 1]. The gate rate depends on the task (see spec section 5).

## What was tested (all on this machine, CPU only)

| Test | Result |
|---|---|
| C++ behavioural tests (determinism, learning, prune count, bounded output, finite under extreme inputs, XOR) | pass |
| NumPy vs C++ parity, 2000 samples, same start | float64 max diff 3.3e-15 (d=8), 1.6e-15 (d=64); float32 1.7e-6, 1.2e-6 |
| CUDA kernel logic on CPU emulator vs C++ | exact (0 difference) for d = 8, 64, 100, several thread counts; float32 about 1e-6 |
| Digits, C++ classifier, one shuffle, 3 passes | 98.5% holdout (599 samples). The C++ uses different random starting hyperplanes from the NumPy runs, so this is one more draw, not a replication of the 97.0% mean over 5 shuffles in the spec |
| Speed, single thread (noisy, about 20%) | d=8, one net: about 11-14 us per sample. d=64, 10 nets: about 410-490 us (float64), 300-330 us (float32). A matched small MLP is about 6-7x faster at d=8 and 17-22x faster at d=64 |

## What was not tested

- **CUDA on a GPU.** The kernel passed a clang device-mode syntax check with stubbed CUDA built-ins and ran correctly on the CPU emulator. Real compilation with nvcc, race behaviour on hardware, and speed are all unknown. The emulator cannot show GPU-specific problems. For one stream at batch size 1 expect latency to be launch-bound; the design targets many networks at once.
- Real-world datasets other than digits; widths above 16; stronger baselines. The spec lists the selection effects behind its numbers.
- Multi-threading and SIMD tuning (roadmap in `docs/TGN_impl_spec.md`).
- Inputs above 128 dimensions (the prune scratch in the kernel is fixed at 128; the C++ library has no such limit).

## Notes

- `docs/TGN_impl_spec.md` embeds the first single-file C++ port; the library in `include/hgln.hpp` is the same algorithm restructured, and passes the same parity tests.
- The competence weighting in the mixture added almost nothing in the ablation (spec section 7b); it is kept for fidelity with the reference.
- No licence file is included; add one before releasing.
