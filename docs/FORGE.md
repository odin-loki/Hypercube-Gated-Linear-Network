# HGLN and FORGE

[FORGE](https://github.com/odin-loki/FORGE) is Odin Loch's C++23 library for running AI models on CPUs and GPUs. It covers runtime SIMD dispatch, GPU runtimes loaded with `dlopen`, a thread pool, NUMA and memory budgets. Both projects are AGPL-3.0-or-later. This page records what HGLN took from FORGE and where FORGE could help further. HGLN itself stays a dependency-free C++17 header, and nothing here makes FORGE a requirement.

## Used now

**Header-free CUDA build with clang (from FORGE's `gpu/` and `ops/gen/gpu_gen.py`).** FORGE builds its CUDA images with

    clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_70|sm_80 -O3 -Werror -S

It needs no CUDA SDK, reads thread IDs through NVVM builtins and makes no libdevice calls. HGLN now does the same in `cuda/hgln_ptx.cu` (`make ptx`), so its kernel compiles to PTX for sm_70 and sm_80 in float64 and float32. The original kernel had never been compiled for a GPU target. The PTX has no external calls, so a CUDA driver can JIT and load it with `cuModuleLoadData`, which is how FORGE loads its own images. Not done: assembling it with `ptxas` and running it on hardware.

## Could use next (not done)

| FORGE part | What it would give HGLN | Notes |
|---|---|---|
| `gpu::cuda_hip` runtime (dlopen of `libcuda`/`libamdhip64`, PTX loading, streams, graphs) | Running the HGLN kernel on a GPU with no nvcc and no CUDA SDK at build time, using the PTX from `make ptx` | The runtime runs FORGE's own spec ops (`run_op`). It does not expose "load this module and launch that symbol", so HGLN needs either a small generic module/launch entry point in FORGE, or a FORGE op for HGLN. FORGE's CUDA path has run on an RTX 3090, so this is the shortest route to a first real GPU test of HGLN |
| HIP image build (`--offload-arch=gfx90a ...`) | An AMD GPU build of the same kernel source | The kernel uses only `__syncthreads`, shared memory and plain arithmetic, so the port should be small. Untried |
| `simd` dispatch (`detect()`, one kernel table per target: scalar, AVX2, AVX-512, NEON, SVE, RVV) | One portable binary that runs the fast path on any CPU | HGLN's speed now depends on `-march`: a plain x86-64 build is correct but about 2.5x slower (OPTIMISATION.md section 4). Compiling `Layer::forward`/`update` once per target and choosing at start-up, as FORGE does, would remove that |
| `simd/math.hpp` (vectorised exp and friends) | A shared, tested vector exp | HGLN's own `hgln_gate()` already vectorises. Using FORGE's would only remove duplicated code |
| `task::pool` (`pool::global()`, lock-free dispatch while workers are awake) | One pool shared with the rest of an application, so HGLN never oversubscribes cores when it is embedded next to FORGE inference | HGLN's `ThreadPool`/`parallel_for` fit the same `(item, worker)` shape, so an adapter is a few lines |
| `os::numa` (node-local allocation, thread binding) and `os::hugepages` | On multi-socket machines, keeping each network's state on the node of the thread that trains it | Matters for the many-network, float64 case, which is partly memory-bound (OPTIMISATION.md section 1) |
| `mem` budgets (`memory.md`) | Admission control: refuse to create more networks than fit | A network's size is known up front (`Net::model_bytes()`) |
| `.fpk` model file (BLAKE3, Ed25519 signing, encryption) | Saving, loading, signing and shipping trained HGLN networks | HGLN has no save/load yet. The `Layer` accessors give everything a serialiser needs |
| `forge-serve` / load balancer | Serving an online learner behind an HTTP API, with predict and learn endpoints | Would need an HGLN session type in FORGE |

The first two rows are the most direct: they would get HGLN onto real GPUs without nvcc.
