# HGLN on FORGE: SIMD and GPU

[FORGE](https://github.com/odin-loki/FORGE) is Odin Loch's C++23 library for running AI models on CPUs and GPUs. HGLN's CMake build uses it for two things:

- **SIMD.** It picks, at run time, the CPU kernel table that suits the machine (scalar, AVX2, AVX-512; NEON, SVE and RVV are also built on those architectures).
- **GPU.** It runs the HGLN kernel through FORGE's CUDA/HIP runtime, with no nvcc, CUDA SDK or ROCm needed at build time.

The header-only library (`include/hgln.hpp`) still works on its own with any C++17 compiler (`make`). FORGE is only needed for the CMake build. Both projects are AGPL-3.0-or-later.

## Build

```sh
# FORGE needs a C++23 compiler (clang 18 or g++ 14), CMake >= 3.25, Ninja, Python 3
CXX=clang++-18 CC=clang-18 cmake -S . -B build/forge -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/forge
ctest --test-dir build/forge          # 100 tests; the GPU mock tests take most of the time
./build/forge/hgln_cli info           # SIMD table in use, CPU features, tables built
./build/forge/hgln_gpu info           # GPU devices, or why there are none
```

How FORGE is obtained (`cmake/hgln_forge.cmake`):

- By default it fetches FORGE at the pinned commit `FORGE_GIT_TAG` (currently `b135941`).
- Alternatively, `-DHGLN_FORGE_SOURCE_DIR=/path/to/FORGE` uses a local checkout.
- Either way FORGE is copied into the build tree, patched there with every file in `third_party/forge-patches/` (the checkout itself is never modified), and added as a subproject with its own tests, apps and examples off.

Targets:

| Target | Contents |
|---|---|
| `hgln` | The header-only library (INTERFACE) |
| `hgln_simd` | The dispatched CPU kernels. Linking it and defining `HGLN_WITH_FORGE=1` routes `Layer::forward`/`update` through the chosen table |
| `hgln_gpu` | `hgln::gpu::Bank`, plus the embedded GPU images |
| `hgln_cli`, `hgln_cli_f` | The CLI (float64 / float32) with SIMD dispatch, plus `info` |
| `hgln_gpu` executable | `info`, `selftest`, `bench` on a GPU |

## SIMD: run-time dispatch (`include/hgln_simd.hpp`, `src/simd/`)

- **What is built per target.** The per-sample kernels (`Layer::forward_k` / `update_k` in `hgln.hpp`) are compiled once per CPU target, each in its own translation unit (`src/simd/kernels_<target>.cpp`). The targets and flags are FORGE's: x86-64 scalar; AVX2 (`-mavx2 -mfma -mf16c`); AVX-512 (`-mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma`); aarch64 NEON and SVE per vector length; riscv64 RVV per VLEN. Every TU adds `-ffp-contract=off`.
- **How the table is chosen.** The table is picked once per process from FORGE's own choice (`forge::simd::active()`). That means FORGE's feature and OS-state checks apply, and `FORGE_SIMD_TARGET=scalar|avx2|avx512|...` forces a table exactly as it does for FORGE. You can also choose in code: `hgln::simd::force("avx2")`, `table(name)`, `targets()` and `install(nullptr)` (back to the automatic choice).
- **Use of FORGE vectors.** The float32 kernels on AVX2, AVX-512, NEON and SVE (up to 512 bits) use FORGE's portable vectors `forge::simd::vec<float, Target>`. They use only add, sub, mul and div, never FORGE's `fma`, so the operation order stays the reference's. FORGE's vectors are float-only, so the float64 kernels use HGLN's own vector code compiled per target. `-DHGLN_SIMD_FORGE_VEC=OFF` uses HGLN's vector code for float32 too.
- **Same results on every table.** All tables give bit-identical predictions and model state, equal to the header-only build. `tests/test_simd.cpp` checks this. It covers d = 3, 8 and 64, float32 and float64, past the prune, comparing predictions, unit outputs, all layer state and the const `infer` path. A deliberately contracted (FMA) AVX2 build makes the test fail, so the check is real.

**Speed** (µs per sample; middle of 3 runs; 4-vCPU AVX-512 Xeon; same session):

| | Original code | Makefile `-march=native` | FORGE dispatch: avx512 | avx2 | scalar |
|---|---|---|---|---|---|
| d = 8, 1 net, float64 | 14.3 | 6.6 | **4.9** | 6.4 | 9.1 |
| d = 64, 10 nets, float64, 1 thread | 492 | 241 | **192** | 220 | 323 |
| same, nets on 4 threads | | 57 | **35** | 56 | 75 |
| d = 64, 10 nets, float32, 1 thread | 334 | 75 | 78 | 81 | |
| same, nets on 4 threads | | 22 | **20** | 27 | |

The dispatched AVX-512 table is the fastest build in float64 and about equal to the native build in float32. A portable binary no longer loses the fast path: the AVX2 table is roughly native speed in float64. Without dispatch, a build with no `-march` ran the d = 8 case at 9 to 13 µs.

Not tested: the NEON, SVE and RVV tables were syntax-checked with clang for aarch64 and riscv64, but not compiled to code or run (no cross toolchain or qemu here). RVV at 128 bits fails inside FORGE's `vec.hpp` with clang 18, as FORGE's own RVV kernels do. The MSVC branch of `cmake/hgln_simd.cmake` is untested.

## GPU: FORGE's CUDA/HIP runtime (`include/hgln_gpu.hpp`, `src/gpu/`)

```cpp
hgln::gpu::Bank<double> bank; std::string why;
if (!bank.open(hgln::gpu::Api::cuda, 0, &why)) { /* no GPU: `why` says why; use the CPU library */ }
bank.upload(nets);              // std::vector<hgln::Net<double>>, same d and Params
bank.run(X, Y, P, T);           // X: T x d shared input stream; Y, P: nets x T; state stays on the device
bank.download(nets);            // back into the CPU networks
```

- **Kernel images.** `cuda/hgln_ptx.cu` is a header-free wrapper around `cuda/hgln_kernel.h`. CMake compiles it with FORGE's clang (no SDK) into:
  - PTX for sm_80 and sm_70;
  - AMDGPU code objects for gfx90a, gfx942 and gfx1100 (FORGE's `FORGE_HIP_ARCHS`).

  Each is built in float64 and float32, with `-ffp-contract=off`. Each image is checked for unresolved symbols, then embedded in the library. If clang lacks a backend, those images are skipped with a message.
- **Runtime.** `Bank::open(Api::cuda | Api::hip, ordinal, &why, Options{threads, scratch})` opens a `forge::gpu::cuda_hip::runtime`, which dlopens `libcuda` or `libamdhip64`. It then loads the first image the device accepts: sm_80 before sm_70, or the matching gfx target. `HGLN_GPU_IMAGE` forces one.
- **Scratch.** The per-sample scratch goes in dynamic shared memory when it fits (CUDA opt-in above 48 KB), otherwise in global memory. `Options::scratch` or `HGLN_GPU_SCRATCH` can force either.
- **FORGE patch.** FORGE's runtime had no public way to load a caller's module, so HGLN carries `third_party/forge-patches/0001-gpu-cuda-hip-caller-modules.patch`. It adds about 45 lines to `forge/gpu/cuda.hpp`: `load_module`, `module_function`, `unload_module`, `max_dynamic_smem`, `set_dynamic_smem` and `launch_kernel`. Each takes the runtime's lock and binds its context. It is written so it can be merged into FORGE as is, after which the patch file can be deleted.

**Tests without a GPU.** `tests/mock_hgln.c` builds stand-ins for `libcuda.so.1` and `libamdhip64.so.6`. They export every entry point FORGE's runtime needs and behave like a driver in these ways:

- they refuse a PTX newer than the mock's architecture, and a code object for another gfx target;
- they enforce the shared-memory limits;
- they run the real kernel source on CPU threads, with one barrier per `__syncthreads` and shared memory pre-filled with NaN.

The ctests drive the full path: CPU `Net` → `Bank` → FORGE runtime → mock driver → download. They cover CUDA and HIP; sm_80, sm_70, gfx90a and gfx1100; d = 8, 64 and 100; shared and global scratch; five calls with state resident, crossing the prune; and a mid-stream download and re-upload. Results:

- float64 predictions and state match the CPU library **exactly** (0 difference);
- float32 matches **exactly** too (0 difference);
- a device with too little shared memory falls back to global scratch;
- a missing driver gives a clear reason.

**Two kernels, one image.** Every image holds `hgln_run`, the float `Net` with every option (4.3, 4.2, and the version 5.0 latch, latch credit, gate error and top-K corners), and `hglnq_run`, the integer `NetQ` (`cuda/hgln_q_kernel.h`), which `hgln::gpu::BankQ` runs. The mock tests cover both (`*_v5*`, `*_int` cases: CUDA and HIP, sm_80 and sm_70, 8/12/16 units, fan-out and chain, several calls, download and re-upload).

**CPU and GPU give the same bits.** Every result is exact (0 difference) against the CPU library, in the mock tests and on hardware:
- the float kernel does the library's IEEE operations in the same order, multiply-adds are never fused, and it uses no libm: the gate (`hgln_gate`) and the `exp` of the latch and the gate-rate schedule (`hgln_exp`) are HGLN's own, and `sqrt` and division are correctly rounded on every device;
- the integer kernel is integer arithmetic, apart from the conversion of inputs and targets and the double estimates inside its exact division and square root.

**Real hardware: an NVIDIA GeForce RTX 3090** (sm_86, CUDA driver under WSL2). FORGE's runtime opens it and the driver JIT-loads the sm_80 PTX. `hgln_gpu selftest` (GPU vs CPU library) gives 0 difference in predictions and state for float64, float32 and the integer net: 4.3, v4.2 and v5, 8/12/16 units, fan-out and chain, d = 8, 64 and 100, up to 4,500 samples (past the prune and the renormalisation at 4,096).

**Not verified:** the AMD code objects have not been loaded by ROCm, and HIP's shared-memory limit is assumed to be 64 KB, not queried. GPU speed has not been studied. With one block per network a single stream takes tens of �s per sample; the kernels are built for many networks at once.

`cuda/hgln_cuda.cu` (the nvcc runtime-API host) is kept as an alternative for anyone with the CUDA SDK. It compiles with nvcc 12.0 (`-fmad=false`, so multiply-adds are not fused) and its selftest gives 0 difference on the RTX 3090 as well (float network only).

## Other FORGE parts that could help later (not done)

| FORGE part | Use for HGLN |
|---|---|
| `task::pool` (one process-wide pool) | Share threads with the rest of an application instead of HGLN's own `ThreadPool` |
| `os::numa`, `os::hugepages` | Keep each network's state on its thread's NUMA node (the many-network float64 case is partly memory-bound) |
| `mem` budgets | Admission control when creating many networks (`Net::model_bytes()` is known up front) |
| `.fpk` model files (signing, encryption) | Saving and shipping trained networks |
| `forge-serve` | An HTTP endpoint for online predict and learn |
| Vulkan / OpenCL / Level Zero / Metal runtimes | More GPU vendors; each needs the kernel in that backend's language |
