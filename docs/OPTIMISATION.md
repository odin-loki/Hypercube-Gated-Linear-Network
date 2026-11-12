# HGLN optimisation notes (version 4.2-opt)

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
| CUDA kernel on the CPU emulator vs the C++ library | float64: exactly 0 difference in predictions and in all live state, for d = 8, 64 and 100, with scratch in global memory and in (NaN-prefilled) shared memory. float32: about 1e-6 |
| CUDA kernel compiled for a GPU | Compiles with clang 18 to PTX for sm_70 and sm_80, float64 and float32, with `-Wall -Werror` and no unresolved external calls (`make ptx`). **Not run on a GPU; not assembled with ptxas.** |

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
- `hgln_cuda.cu` now frees its device memory. It is still **not compiled** (no nvcc here).

The emulator (`cuda_emu d nets T threads [shared]`) now exits non-zero on a mismatch. In shared mode it fills the scratch with NaN first, so any read of scratch before it is written would show up as a difference.

## 4. Not done, and ideas not tried

- **Runtime CPU dispatch.** The fast path depends on the compile flags. A build without `-march` (plain SSE2) is correct but runs at the original speed (about 15 us at d = 8). `make ARCH=-march=x86-64-v3` gives a binary for any AVX2 machine. FORGE's `simd` layer picks kernels at run time, and so could HGLN (see `FORGE.md`).
- **FMA contraction.** `-ffp-contract=fast` was about 20% faster at d = 64 in one (noisy) measurement. It changes the last bits relative to the reference, and would break the exact library-vs-kernel agreement unless both are built that way, so it is off.
- **Splitting one network across threads.** One network's sample costs 2.5 to 15 us, and the layers depend on each other, so a hand-off per layer would cost about as much as it saves. Only independent networks are parallelised.
- **GPU.** Nothing has run on a GPU. Real-hardware parity, race behaviour and speed are unknown, as before.
- **Skipping negligible corners** (roadmap item 6 of the implementation spec) changes the algorithm and was not attempted.
- The NumPy reference (`python/hgln_ref.py`) is unchanged: it is the oracle the C++ is checked against.

## 5. How to reproduce

    make && make test      # behavioural, parity, emulator (global + shared, float64 + float32), PTX, digits
    make bench             # the speed table above (this machine's numbers will differ)
