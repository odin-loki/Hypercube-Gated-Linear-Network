# HGLN optimisation notes (version 4.2-opt, and the version 5.0 integer net in section 5)

> **Update:** the CMake build now uses FORGE for run-time SIMD dispatch (faster than the native build below: 4.9 µs at d = 8, 192 µs at d = 64 with 10 nets) and for GPU execution through FORGE's CUDA/HIP runtime. See [FORGE.md](FORGE.md).

**Author:** Odin Loch. Licence: AGPL-3.0-or-later.

This page records what was changed to make the C++ library and the CUDA kernel faster, more parallel and smaller in memory. It covers how each change was checked, what was measured, and what was not done. The algorithm is unchanged: it is still the procedure in `TGN_impl_spec.md` section 4.

## 1. Results

**Machine.** 4 vCPU cloud VM, Intel Xeon @ 2.1 GHz with AVX-512, 2 MB L2 per core, shared with other tenants. g++ 13.3, `-O3 -march=native -std=c++17`. The original code was rebuilt and run in the same session, so the two columns compare like with like. Each figure is the middle of three runs. Run-to-run noise is about 10 to 20 percent.

### Speed (microseconds per sample; `hgln_d|hgln_f bench d nets N [threads]`)

| Setting | Original | Now, 1 thread | Now, all nets in parallel (4 threads) | MLP 3x16 |
|---|---|---|---|---|
| d = 8, 1 net, float64, predict + learn | 14.3 | **6.0** (2.4x) | 6.0 (one net cannot be split) | 2.0 |
| d = 8, 1 net, float64, predict only | 7.4 | **2.65** (2.8x) | 2.5 | |
| d = 64, 10 nets, float64, predict + learn | 492 | **246** (2.0x) | **52** (9.4x) | 26 |
| d = 64, 10 nets, float64, predict only | 251 | 130 | 20 | |
| d = 64, 10 nets, float32, predict + learn | 334 | **72.5** (4.6x) | **22.6** (14.8x) | 26 |
| d = 64, 10 nets, float32, predict only | 193 | 36 | 8.6 | |

- The gap to the matched MLP has narrowed. Single-threaded it was 7x at d = 8 and 13 to 19x at d = 64. It is now 3x at d = 8, 9x at d = 64 in float64 and 2.8x at d = 64 in float32.
- With 10 nets at d = 64 in float64 the working set (10 x 278 KB) is larger than one core's L2, so that row is partly memory-bound. Float32 halves the bytes, which is why it gains much more than float64. Spreading the nets over cores also spreads them over the cores' caches, which is why 4 threads give more than 4x.
- **Per-sample threading.** `Classifier::set_threads(4)` runs a single `train()` call's heads on a persistent pool. At d = 64 with 10 heads, float64, averaged over the first 2,000 samples (before and after the prune), it measured 290 us with 1 thread, 147 with 2 and 107 with 4. At d = 8 it gains little (70 to 60 us), because each head's work is too small to split.

### Memory (bytes per network)

| | Original | Now, at start | Now, after the prune (sample 512) |
|---|---|---|---|
| d = 8, float64 | 149,296 | 137,088 | **99,904** (-33%) |
| d = 64, float64 | 417,424 | 371,328 | **278,400** (-33%) |
| d = 64, float32 | 209,588 | 189,348 | **141,668** (-32%) |

"Now" is the model (`Net::model_bytes()`) plus the per-network scratch (`Work`). For `infer()` from several threads you add one `Work` per thread (about 8 to 17 KB), not one per network.

### Accuracy and agreement (all re-run after the changes)

| Check | Result |
|---|---|
| NumPy vs C++ parity, 2,000 samples, through the prune | float64 2.9e-15 (d = 8), 1.8e-15 (d = 64); float32 1.3e-6, 1.2e-6 (tolerances 1e-12 / 1e-4). Same with clang++ 18, with no `-march`, and with the plain-C++ fallback (`-DHGLN_NO_VECTOR_EXT`) |
| Digits, C++ classifier, one shuffle, 3 passes | 95.8 / 98.3 / 98.5% holdout, identical to the original build |
| Threaded vs serial | `train_batch`, pooled `train()`, `predict_batch` and `run_streams` give bit-identical models and outputs to the serial calls (tests in `tests/test_api.cpp`) |
| CUDA kernel on the CPU emulator vs the C++ library | float64: exactly 0 difference in predictions and in all live state, for d = 8, 64 and 100, with scratch in global memory and in (NaN-prefilled) shared memory. float32: about 1e-6 (version 4.2-opt measurement; *since then float32 is exact too, 0 difference*) |
| CUDA kernel compiled for a GPU | Compiles with clang 18 to PTX for sm_70 and sm_80, float64 and float32, with `-Wall -Werror` and no unresolved external calls (`make ptx`). **Not run on a GPU; not assembled with ptxas.** *Since run on an RTX 3090: 0 difference (`FORGE.md`).* |

## 2. What changed in the C++ library (`include/hgln.hpp`)

Each change either keeps every floating-point operation in the same order as the reference (exact), or is marked.

1. **SIMD layouts, lane = unit.** Everything per unit (the 64 gate dot products, the sigmoids, the 16-corner heatmap, the mixture, the competence map, the gate gradients, the LMS rate) is now computed for all 16 units of a layer at once, one unit per vector lane. `H` is stored `[input][hyperplane][unit]`, `p` as `[input][unit]` and `m` as `[corner][unit]`. Each lane does exactly the scalar reference's arithmetic in the same order, so the results are unchanged. *Exact.*
2. **Rule dot products, lane = corner.** `W` is stored `[unit][column][corner]`. The 16 rule outputs of a unit are one 16-wide accumulator: each input is broadcast and multiplied by a row of 16 corner weights. The sum over inputs stays in ascending order. *Exact.*
3. **Prune compaction.** At sample 512 each unit keeps 60% of its inputs and the rest are masked to zero for ever. The masked columns are now removed from `W` (an index list `cols` replaces the mask `M`), so the rule loops do about 40% less work and `W` is about 40% smaller. Adding the masked zero terms never changed a sum, so the results are unchanged. *Exact.*
4. **Dead state freed.** The activity trace `A` only feeds the prune, and after the one-off prune nothing reads it, so it is freed and no longer updated. Later prune calls were already no-ops and now return at once. *Exact.*
5. **`pu = p * u` recomputed, not stored.** The p step and the hyperplane step are fused into one pass over the inputs, which reads the old `H` and old `p` exactly as the reference does. This saves 16 x D values per layer and a pass. *Exact.*
6. **Gate without libm.** `hgln_gate()` (`include/hgln_gate.h`) computes `clip(1/(1+exp(-12z)), 1e-4, 1-1e-4)`. Outside `|12z| <= 9.3` the clip always wins. Inside that range exp uses Cody-Waite range reduction, a degree-13 polynomial and an exactly built power of two. It has no branches and no libm calls, so 64 gates are a few vector instructions. Measured against `1/(1+std::exp(x))` over the gate range: largest relative difference 3.1e-16 in float64 (about 1 ulp) and 1.7e-7 to 6.6e-7 in float32 (a few ulp; the reference `expf` differs by build). *Not bit-identical to `std::exp`, but well inside the parity tolerance.*
7. **Renormalisation multiplies by 1/|H|** instead of dividing by |H| (D x 64 divisions per layer were the single largest cost of `learn`). This is at most 1 ulp different per element. *Not bit-identical to the reference's division.* The CUDA kernel does the same, so the kernel and the library still agree exactly.
8. **Aligned, allocation-free buffers.** All buffers are 64-byte aligned and allocated in `init()`. Nothing is allocated per sample. The only allocation after `init()` is the one at the prune, which shrinks `W`.
9. **Const inference.** `Net::infer(x, Work&) const` predicts without touching the model. `predict()` is `infer()` with the network's own `Work`. Any number of threads can call `infer()` on one shared network, each with its own `Work`.
10. **Parallel APIs.**
    - `hgln::parallel_for(n, threads, fn)`: dynamic scheduling, with the calling thread taking part.
    - `hgln::ThreadPool`: persistent workers for per-sample dispatch. Workers spin briefly, then sleep.
    - `Classifier::train_batch` / `predict_batch` for blocks of samples, and `Classifier::set_threads` for single-sample calls.
    - `hgln::run_streams(nets, X, Y, P, N, threads)`: many networks on one input stream, the CPU counterpart of the CUDA kernel.
    - Online semantics are kept: every network still learns sample by sample in order. Only independent networks run concurrently, so results do not depend on the thread count.

Portability: the vector code uses GCC/Clang vector extensions. Other compilers (e.g. MSVC) get a plain-array fallback with the same semantics, which also passes parity. The fast exp's rounding trick assumes no excess precision (`FLT_EVAL_METHOD == 0`, true on x86-64, ARM64 and RISC-V). Elsewhere it falls back to `std::nearbyint`.

## 3. What changed in the CUDA kernel (`cuda/`)

- **Scratch in shared memory.** Pass `scratch = nullptr` and launch with `hg_dims(d).scs * sizeof(hreal)` bytes of dynamic shared memory: 23 KB at d = 8, 31 KB at d = 64 and 36 KB at d = 100 in float64, half that in float32. `hgln_cuda.cu` does this when it fits (opting in above 48 KB) and falls back to global memory otherwise. The global scratch allocation (one per network) disappears.
- **One phase and 48 x D scratch values fewer.** The `p` step and the hyperplane step are fused per (unit, input) element, so the stored `pu` and a barrier are gone.
- **Dead work skipped.** The activity trace and the prune scan stop after the prune time `hg_prune_time(warm)` (sample 512).
- **Same gate and renormalisation as the C++ library** (`hgln_gate()`, multiply by 1/|H|), so the emulator still agrees exactly.
- **Builds without the CUDA SDK.** `cuda/hgln_ptx.cu` wraps the kernel in a header-free shim and `make ptx` builds PTX with plain clang. It uses the recipe FORGE uses for its GPU images (see `FORGE.md`). The shim supplies thread IDs through NVVM builtins and its own `exp`, so the PTX has no libdevice calls and a driver can JIT-load it directly.
- `hgln_cuda.cu` now frees its device memory. It was then still **not compiled** (no nvcc here). *Since compiled with nvcc 12.0 and run on an RTX 3090: its selftest gives 0 difference.*

The emulator (`cuda_emu d nets T threads [shared]`) now exits non-zero on a mismatch. In shared mode it fills the scratch with NaN first, so any read of scratch before it is written would show up as a difference.

## 4. Not done, and ideas not tried

- **Runtime CPU dispatch.** *Now done with FORGE, see `FORGE.md`; this item describes the header-only build.* The fast path depends on the compile flags. A build without `-march` (plain SSE2) is correct but runs at the original speed (about 15 us at d = 8). `make ARCH=-march=x86-64-v3` gives a binary for any AVX2 machine. FORGE's `simd` layer picks kernels at run time, and so could HGLN (see `FORGE.md`).
- **FMA contraction.** It is now explicitly off everywhere (`-ffp-contract=off` in the Makefile, in every SIMD and GPU translation unit, and in the PTX/AMDGPU builds). Correction: g++ defaults to `-ffp-contract=fast` for C++ even with `-std=c++17`, and clang's CUDA mode fuses multiply-adds too. So the measurements in section 1 (Makefile, g++ `-march=native`) were taken with FMAs fused, and the first PTX contained FMAs. With contraction off, every build (g++ or clang, any `-march`, any SIMD table, the fallback path) gives the same output bits.
- **Splitting one network across threads.** One network's sample costs 2.5 to 15 us, and the layers depend on each other, so a hand-off per layer would cost about as much as it saves. Only independent networks are parallelised.
- **GPU.** *Since done:* both kernels (float and integer) have run on an NVIDIA RTX 3090 and reproduce the CPU library bit for bit, float64, float32 and int16 (see `FORGE.md`). *Since studied: section 6.*
- **Skipping negligible corners** (roadmap item 6 of the implementation spec) changes the algorithm and was not attempted here. *Since done as top-K corners (`Params::corners`, default 6 in version 5.0, `ALGORITHM.md` section 8a).*
- The NumPy reference (`python/hgln_ref.py`) is unchanged: it is the oracle the C++ is checked against.

## 5. The version 5.0 integer net (`include/hgln_q.hpp`)

**Machine.** WSL2 on a 32-core Xeon Gold 6242 (AVX-512 with VNNI), g++ 13.3, `-O3 -march=native -ffp-contract=off`, one thread. The host was about 60% busy with unrelated jobs, so single runs varied by up to ±40%. Every figure below is the **minimum over 25 rounds** in which all versions ran alternately on rotating cores (`qbench`: v5 settings, 1 net at d = 8, 4 nets at d = 64, past the prune). The float32 rows, identical code in every round, varied by about ±5%.

| µs per sample (predict + learn) | d = 8 | d = 64 | Bytes per net after the prune (d = 8 / 64) |
|---|---|---|---|
| float32 `Net`, 4.3 settings, 8 units | 2.49 | 4.9-5.2 | 16,664 / 61,496 |
| float32 `Net`, v5 settings but all 16 corners (measured before the float `Net` had top-K) | 3.9-4.1 | 6.5-6.9 | 16,664 / 61,496 |
| int16 `NetQ`, stochastic rounding (before) | 14.7 | 32.9 | 8,672 / 31,872 |
| int16 `NetQ`, round to nearest | 7.3 | 14.7 | 8,624 / 31,824 |
| **int16 `NetQ`, now** | **5.95** | **13.4** | **8,624 / 31,824** |

The "before" row is from single runs. Every change below is exact: all CPU targets (scalar, SSE2, AVX2, AVX-512 with and without VNNI, clang and g++, the plain-C++ fallback) stay bit-identical to the research reference (`tests/test_q.cpp`).

1. **No random numbers.** Round-to-nearest-even replaces Philox stochastic rounding (the accuracy study is in `ALGORITHM.md` section 8c). This removed about half of `learn`.
2. **Rule-step scalars across units.** The squared heatmap norm, the LMS denominator, the exact division and the 16 rule steps of every unit are computed with one unit per lane, through the same exact vector division the gate steps use, instead of per unit in scalar code.
3. **Rule-weight update in double precision.** `rne(sc x, F)`: `|sc x| < 2^46` is exact in a double, the scaling by `2^-F` is exact, and an AVX-512 conversion with embedded round-to-nearest-even gives exactly the spec's `rne`. This replaces a 64-bit multiply, a 5-instruction rounding and a narrowing per element.
4. **Top-K threshold with a compile-time sorting network.** Batcher's odd-even merge sort (63 compare-exchanges) is unrolled at compile time, so the 16 keys stay in registers; the old version read the pairs from a table and kept the keys on the stack.
5. **Gate dot products with `vpdpwssd` / `vpmaddwd`.** `|H|` and the gate inputs fit in int16 and the spec's sums wrap mod 2^32, so inputs i and i+1 are interleaved and accumulated as int16 pairs. This cut prediction at d = 64 by about 25% (4.6 to 3.5 µs).
6. **Fixed:** the plain-C++ fallback (`-DHGLN_NO_VECTOR_EXT`, the path MSVC would take) did not compile; scalar operands of mixed type now broadcast as they do with the vector extension.

**Tried and dropped** (slower in the alternating runs): keeping the hyperplane scratch as int64 to avoid narrowing and widening (+1 to +2 µs, from the doubled memory traffic); one shared reciprocal for the latch's two divisions together with skipping the "keep H when its norm is 0" blend (+0.5 to +2 µs).

**Earlier conclusion (before the mixed datapath).** The goal was the integer net at least as fast as float32 at 8 units. With exact integer learning it stayed 2.4x (d = 8) and 2.6x (d = 64) slower than float32 with the 4.3 settings. The main cost was the gate-weight / hyperplane update and its exact renormalisation, whose 64-bit intermediates (`cb x p x s` reaches 2^55) run 8 lanes per AVX-512 register where float32 runs 16. Closing the gap needed a spec change.

### 5b. Mixed float / integer learning (the current spec)

The integer net keeps its int16 state and its integer forward pass, which is already about as fast as float32. Learning is now mixed (`hgln_qspec.hpp` section 6, accuracy in `ALGORITHM.md` section 8c):
- **Per-element updates in float32.** The gate-weight, hyperplane (with its renormalisation: one `sqrt` and one division per row) and rule-weight updates are float32 on 16 lanes instead of int64 on 8, rounded back to int16 with `(v + 1.5 2^23) - 1.5 2^23`.
- **Three control-scalar divisions in double.** The latch gradient, its credit and the LMS rate use one IEEE double division rounded to nearest, instead of the exact integer division with correction steps.
- **Implementation fixes found while profiling:** integer clamps now compile to min / max instructions instead of compare-and-select, and the renormalisation skips its "keep H" blend when no row needs it.

The float `Net` was profiled at the same time. The latch's `hgln_exp` now runs in simple loops that vectorise (float32 v5 prediction 2.1 -> 1.3 us at d = 8), and the top-K ranking is a branch-free vector pass over the 120 corner pairs (1.3 -> 0.15 us per layer; the first, branchy version cost about 20 us per sample).

Speed, best of 12 alternating rounds (`qbench`: one net at d = 8, four at d = 64, past the prune; µs per sample, predict + learn):

| | d = 8, g++ | d = 8, clang 18 | d = 64, g++ | d = 64, clang 18 |
|---|---|---|---|---|
| float32 `Net`, 4.3 settings, 8 units | 2.03 | 2.21 | 4.46 | 4.44 |
| float32 `Net`, v5 defaults (top-6 corners) | 3.12 | 3.54 | 5.57 | 5.73 |
| int16 `NetQ`, exact integer learning | 5.77 | 5.09 | 12.01 | 11.03 |
| **int16 `NetQ`, mixed learning (version 5.0)** | **5.34** | **4.73** | **10.89** | **10.08** |

**Result.** The integer net is 1.3-1.7x (d = 8) and 1.8-2.0x (d = 64) slower than float32 running the same v5 model, while using half the memory and giving the same bits on every CPU and GPU. It is not at parity. Most of what remains is the per-element update work: float32 updates a value in place, while the integer net converts int16 to float, rounds and clamps back, and makes a second pass for the exact per-row renormalisation. Further steps that would change the spec again: norm tracking instead of per-sample renormalisation (as float 4.3 does), and processing two inputs per register for the 8-unit layers (that changes the summation order of the row norms).

## 6. GPU speed of the float kernel (version 5.0)

Measured on an RTX 3090 (82 SMs, 6 MB L2) with `hgln_gpu bench d nets T`, float32, the v5 model with 8 units and fan-out (the character LM's heads). Times are per sample for all networks together, copies included.

Every step below keeps the kernel **bit-identical** to the CPU library. On the real GPU, predictions and state differ by 0 in 78 runs:
- float64 and float32, both kernels;
- versions 4.3, 4.2 and v5; 8, 12 and 16 units; chain and fan-out;
- d = 8, 64 and 128; 2, 37 and 256 networks.

The CPU emulators also give 0 difference.

| µs per sample | 27 nets, d = 108 | 256 nets, d = 128 | 1024 nets, d = 64 |
|---|---|---|---|
| block kernel, before | 77 | 809 | 954 |
| warp kernel, first version | 238 | 1497 | 2659 |
| block kernel, now (automatic threads) | 41-49 | 178-196 | 357 |
| warp kernel, now | 144-158 | 263-289 | 409 |
| CPU, same heads (`run_streams`, 27 / 32 threads) | 24 | about 200 | - |

**What it was.** A per-phase cycle profile (thread 0 timing every phase with `clock64`) showed:
- **State reads not coalesced.** Each task walks its own row of `H` or `W` in the library's summation order, so a warp's 32 loads touched 32 different cache lines.
- **One thread prepared the next input alone.** That was about 17% of the time at 27 networks, stalling on every store-then-load.
- **The latch / top-K phase ran on 24 threads.** It had one task per unit, so 24 threads did the work of the 96 gates.
- **The kernel body was not inlined.** clang compiled it as a called function with 255 registers and a 1.7 KB stack.

**What changed** (`cuda/hgln_kernel.h`):
1. **Transposed device layout.** `H`, `p`, `W`, `M`, `A` and the scratch `xm` are stored transposed per network (`hgln_dims.h`: `hg_relayout`; the host converts on upload and download). Each task still sums its row in the library's order, but adjacent threads now read adjacent addresses.
2. **Parallel input preparation.** Every thread computes the input norm itself, with the same loop and so the same bits, and the threads share the copy.
3. **Latch in its own phase.** The latch runs one task per gate; then the heatmap and top-K run per unit. This costs one more barrier.
4. **Norm computed in the rule loop.** `|xm|^2` is computed in the rule dot product that already walks `xm`, not again in U1.
5. **Loads issued ahead.** The rule loops load 8 elements ahead, and U2a takes 4 tasks per thread.
6. **Forced inlining.** The kernel body is `always_inline`; the rare prune stays out of line.

**Why the GPU is still not much faster than the CPU.** Every sample reads and writes the whole state of every network: about 115 KB (d = 108) to 135 KB (d = 128) each in float32, for 1-2 floating-point operations per byte.
- **Few networks (one per SM):** each sample is a chain of L2 round trips. The 109-long dot products must be summed in order to keep the bits, so they cannot be split across threads.
- **Many networks:** 256 networks are 35 MB, which does not fit in the L2. The run streams about 69 MB per sample through DRAM, measured at about 350 GB/s.
- **The CPU** keeps each network in its core's cache.

A large GPU speed-up needs a kernel that keeps each network's state on the chip (shared memory plus registers) for the whole stream. At d = 108 the state is larger than the 99 KB of shared memory a block can have, so it would need a split: `H` and `p` in shared memory, `W` from L2. That has not been built.

**The warp kernel** (`hgln_wrun`: one 32-lane warp per network, several networks per block, warp barriers) is correct, and is kept as `Options::kernel = Kernel::warp`. It was never faster than the block kernel, from 27 to 16384 networks:
- with few networks, 32 threads cannot keep enough loads in flight to stream a network's state;
- with many networks, a block kernel with 32 threads per block runs the same schedule and was as fast or slightly faster.

So `Kernel::automatic` is the block kernel, and its thread count is automatic (`Options::threads = 0`): 256 for fewer than 128 networks, otherwise 64. Measured on the 3090:
- 27 networks: 256 threads 50 µs, 128 threads 57 µs, 64 threads 84 µs.
- 256 networks: 64 threads 195 µs, 256 threads 273 µs.

## 7. How to reproduce

    make && make test      # behavioural, parity, emulator (global + shared, float64 + float32), PTX, digits
    make bench             # the speed table above (this machine's numbers will differ)
