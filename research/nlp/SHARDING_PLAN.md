# Plan: sharded training and recombination of HGLN character models

## Goal

Two parts:
1. **Train faster** by splitting the training text into shards and training one model per shard, all at the same time. The shards are independent, so this fills the GPU (thousands of networks instead of 27) and scales across GPUs, CPUs and machines.
2. **Rebuild one model** from the shard models by competitions between their units, so the result is as good as or better than a single model trained on all the data.

Both need to be proven on text8 before scaling to enwik8 and then to 1 TB.

## Why it should work

- **One model stops improving early.** On text8 it reaches 3.52 test BPC after 10M characters and 3.49 after 90M. It is limited by its size, not by data. Each 10M-character shard model should therefore be nearly as good as the 90M model, and combining them is how the extra data gets used.
- **Layer-1 units can be swapped.** A network's prediction is the plain average of its 3U units. Each unit is a local expert: 4 gating hyperplanes on the input choose among 16 rule vectors. A layer-1 unit reads only the raw input, so it can move between networks unchanged, and the order of units does not matter.
- **Layers 2 and 3 cannot.** They read the layer below by position, so after recombination they need a short retrain.
- **Sharding fills the GPU.** With the current kernel on an RTX 3090:

  | Networks at once | Head-steps per second |
  |---|---|
  | 27 | ~0.6M |
  | 256 | ~1.3M |
  | 1024 | ~2.9M |

  The CPU (all threads) does ~1.1M.

## Baselines already measured (text8: 90M train / 5M validation / 5M test, K=4, 27 heads, v5 with 8 units)

| Model | Test BPC | Test accuracy |
|---|---|---|
| HGLN, 10M training characters | 3.522 | 37.7% |
| HGLN, 90M training characters | 3.490 | 37.4% |
| 5-gram / 6-gram (interpolated absolute discounting) | 1.975 / 1.802 | - |

## Phase 0: prerequisites

1. **Commit today's GPU work.** The transposed device layout, parallel input preparation, the latch phase, `|xm|^2` in the rule loop, forced inlining, the warp kernel, and automatic kernel and thread choice. All 104 CMake tests and `run_tests.sh` pass, and the 3090 gives 0 difference in 78 runs.
2. **Calibration.** Fit a scale and bias per class, `p(c) ~ exp(a_c s_c + b_c)`, on the first 1M validation characters (the patch is drafted in `charlm_hgln.cpp`). Report it next to the single temperature. Without it, BPC mostly measures calibration and the combinations below cannot be compared fairly.
3. **Online test score.** Add an option to score the test split while the model keeps learning (predict, then learn). Report it next to the frozen-model score; it explains the 51% online vs 37% frozen accuracy gap.

## Phase 1: the sharded trainer

**Kernel: one input stream per shard.** Today every network in a `Bank` reads the same input stream X. Add `xgroup` (networks per input stream): network n reads `X + (n / xgroup) * T * d`, and `xgroup = nets` keeps today's behaviour. That is one kernel argument and one `Bank::run` overload. Bit-exactness is unaffected, and the emulator, mock and real-GPU checks are extended to grouped input.

**Trainer.** `charlm_shard FILE text8 S [mode gpu|cpu|both] [chunk]`:
- **Shards.** Split the 90M training characters into S contiguous shards, so each shard keeps its text in order.
- **Networks.** S × C networks: shard s, class c is network s·C + c, all trained together.
- **Chunks.** Each pass takes `chunk` characters from every shard, for X of S × chunk × d floats. Chunk 20k at S = 36 is about 310 MB.
- **CPU path.** `run_streams` over the same S × C streams.
- **Both together.** A share of the shards goes to the GPU and the rest to the CPU, both run concurrently, and the models are merged at the end.
- **Output.** The S shard models (`save`), plus timing as characters per second, head-steps per second and the GPU / CPU split.

## Phase 2: combination methods (validation on its first 1M characters; test set only for the final numbers)

| Method | What it is |
|---|---|
| **A. Best single shard** | The best of the S models |
| **B. Ensemble average** | Average the S models' calibrated log-probabilities (or probabilities; try both). The upper reference; costs S× at inference |
| **C. Unit recombination** | One network of normal size built from the S models' units (below) |
| **D. Competition rounds** | C repeated with fresh shards (below) |

**C. Unit recombination**, per head c:
1. **Pool.** The S × U layer-1 units of all shard models for head c.
2. **Select.** Greedy forward selection (ensemble selection) of U units: at each step add the unit that most lowers the validation squared error of the average of the selected units' predictions against the ±1 target. Variants: with replacement; growing U to 16 (the kernel's maximum).
3. **Upper layers.** Initialise layers 2 and 3 from the shard model that contributed the most units, or from scratch.
4. **Retrain.** Briefly retrain on a held-out slice of training text (1-2M characters): the whole network, and a variant that freezes layer 1.
5. **Library support.** `Layer::copy_unit(const Layer& src, int k_src, int k_dst)` copies H, Q, p, W, M, A and the competence map, with a library test that a copied unit predicts identically.

**D. Competition rounds** (population-based):
1. Train from C's network on S new shards.
2. Recombine by C again.
3. Repeat for R rounds.

It costs the same per round as phase 1. Measure BPC per round.

## Phase 3: experiments (text8)

| Run | Shards | Characters per shard | Networks | Estimated GPU time (3090) |
|---|---|---|---|---|
| S = 9 | 9 | 10M | 243 | ~35 min |
| S = 36 | 36 | 2.5M | 972 | ~15-20 min |
| S = 9, CPU + GPU | 9 | 10M | 243 | measure |

For each run, report:
- test BPC (temperature and calibrated), accuracy, and the online score;
- for A-D, wall time, characters per second, head-steps per second.

Write everything to `research/nlp/results_shard.jsonl`. Also generate a text sample from the best single network.

## Decision points

| Question | Success if | Then |
|---|---|---|
| Does sharding speed things up? | S = 36 on the 3090 gives ≥ 2× the CPU's characters per second for the same total data | Move to enwik8 |
| Does recombination work? | C, after the retrain, is within 0.02 BPC of the single 90M model, or better | Run D |
| | C is far behind B | Try a recombined network of 16 units per layer, or distil B into one network |
| Do competitions help? | D improves BPC over rounds | Use D for the 1 TB plan |

## Phase 4: scale-up

1. **enwik8.** 256 heads, d = 128, the same pipeline.
2. **Throughput model for 1 TB.** Characters per second per GPU and per CPU, from phase 3, times the number of devices. Estimate: ~4 months on one 3090 at ~100k characters per second for 27-head models; under 2 weeks on 10 devices.
3. **Kernel option.** The on-chip state kernel (`docs/OPTIMISATION.md` section 6) raises the per-GPU ceiling; sharding works with it unchanged.

## Files

| File | Change |
|---|---|
| `cuda/hgln_kernel.h`, `include/hgln_gpu.hpp`, `src/gpu/hgln_gpu.cpp` | `xgroup` |
| `cuda/cuda_emu.cpp`, `cmake/hgln_gpu.cmake`, `src/gpu/hgln_gpu_cli.cpp` | Grouped-input exactness tests |
| `include/hgln.hpp` | `Layer::copy_unit`; test in `tests/test_api.cpp` |
| `research/nlp/charlm_hgln.cpp` | Calibration; online score |
| `research/nlp/charlm_shard.cpp` | The sharded trainer and combination methods (new) |
| `cmake/hgln_gpu.cmake` | `hgln_charlm_shard` target |
| `research/nlp/results_shard.jsonl`, `research/nlp/README` or `docs/` | Results and write-up |
