# HGLN algorithm changes: versions 4.3 and 5.0

**Author:** Odin Loch. Licence: AGPL-3.0-or-later.

This page asks whether HGLN's mathematics can be made cheaper in operations and memory, and more parallel, without losing accuracy. Every candidate was measured on the benchmarks in `TGN_spec.md`. Two changes passed and became version 4.3; the rest cost accuracy and were not adopted. Version 5.0 (section 8) then takes the cheapest usable structure and adds three learning-rule changes and an integer datapath. All numbers below can be regenerated with `research/alg/` (sections 7 and 8).

## 1. What changed in 4.3

### 1a. Exact norm tracking for the hyperplanes (same model, less work)

Each unit's 4 hyperplanes are unit vectors. Version 4.2 updates them on every sample in three steps:

- add the step `cb_b * pu`, where `pu = p * u`;
- compute the new length, a pass over the D inputs;
- divide by it, a second pass over the D inputs.

Version 4.3 stores each hyperplane **unnormalised**, as `H_b`, together with its squared length `q_b`. Because the step direction is known, the new length follows in closed form:

```
z_b   = (H_b . pu) / |H_b|                      forward pass (one multiply by the stored 1/|H_b|)
s_b   = cb_b * |H_b|                            the same step, scaled to the stored length
H_b  <- H_b + s_b * pu                          the same direction as normalise(H_b/|H_b| + cb_b * pu)
q_b  <- q_b + 2 s_b |H_b| z_b + s_b^2 |pu|^2    since H_b . pu = |H_b| z_b, and |pu|^2 is summed in the forward pass
```

The p step uses the unit hyperplane `H_b / |H_b|`, as before. In exact arithmetic this is the same model, sample for sample. Two passes over the inputs per hyperplane disappear, and on the GPU a whole barrier phase goes with them.

Bounding rounding drift:
- every 4,096 samples each hyperplane is rescaled to unit length and `q` recomputed exactly;
- the same happens at once if any `q` leaves (1e-6, 1e6).

Measured agreement:
- **Experimental code:** with versus without norm tracking, predictions differ by at most 2.3e-15 over 3,000 samples, and every benchmark score in section 3 is identical.
- **Library:** parity with the NumPy reference, which still renormalises, is 2.0e-15 to 3.6e-15 in float64 and 1.0e-6 to 2.0e-6 in float32.

### 1b. The mixture is a plain average (competence map dropped by default)

Version 4.2 weighted each unit's prediction by a learned competence map: 16 values per unit, updated every sample. The 4.2 ablation found that this "adds almost nothing", but it was checked on the synthetic tasks only. It has now been checked on digits too, and over three independent sets of random starting seeds (section 3). Version 4.3 therefore uses equal weights.

The global error signal simplifies to:

```
out  = (1/48) sum_i o_i
es_i = (1 - glob) (y - o_i) + glob (y - out)        (the 4.2 factor r_i / sum_j r_j^2 is exactly 1 for equal weights)
```

This removes the competence map (768 values per network), its per-sample update, the confidence term in the forward pass and the weight normalisation. `Params::competence = true` (or `Params::v42()`) gives the 4.2 mixture exactly. Both modes are parity-tested against the NumPy reference (`python/parity.py ... v42|v43`).

## 2. Benchmarks and method

- **Synthetic drifting streams** A, B and C from `research/t7.py` and `t24.py`: 8 inputs, 4,096 samples, the function changes at sample 2048, seeds 200-209. The score is prequential normalised MSE (lower is better) in the late pre-drift window (1536-2048, "pre") and the late post-drift window (3584-4096, "post").
- **Digits:** one-vs-rest, shuffles 10-14, 1,198 train / 599 test, gate rate 0.02, holdout % after 3 passes.
- **Implementation:** each variant runs in `research/alg/hgln_var.hpp`, a readable scalar implementation with switches. With no switches it matches the library's 4.2 to 3.8e-15.
- **Wins:** "(w/10)" counts the seeds on which the variant beats the 4.2 baseline, paired by seed; "(w/5)" does the same over shuffles.
- **Noise level:** the two "noise" rows rerun the unchanged 4.2 algorithm with different random starting hyperplanes. They show that differences of about ±0.005 NMSE and ±0.5 digits points are noise.
- **Cost columns:** multiply-adds per sample after the prune and persistent state values, counted from the procedure, relative to 4.2.

## 3. Results

Variants marked * also include the plain-average mixture (1b).

| Variant | A pre | A post | B pre | B post | C pre | C post | Digits 3 passes | Ops d=8 | Ops d=64 | State d=64 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4.2 baseline | 0.130 | 0.083 | 0.129 | 0.158 | 0.154 | 0.207 | 96.63 | 1.00x | 1.00x | 1.00x |
| 4.2, other seeds (noise) | 0.127 (7/10) | 0.086 (3/10) | 0.132 (4/10) | 0.156 (7/10) | 0.149 (7/10) | 0.214 (4/10) | 97.16 (4/5) | | | |
| 4.2, other seeds (noise) | 0.124 (7/10) | 0.083 (4/10) | 0.130 (5/10) | 0.160 (8/10) | 0.148 (9/10) | 0.210 (3/10) | 97.13 (4/5) | | | |
| **norm tracking (1a)** | 0.130 (=) | 0.083 (=) | 0.129 (=) | 0.158 (=) | 0.154 (=) | 0.207 (=) | 96.63 (=) | 0.93x | 0.87x | 1.01x |
| **plain average (1b)** | 0.130 (5/10) | 0.077 (9/10) | 0.126 (9/10) | 0.158 (5/10) | 0.155 (2/10) | 0.209 (5/10) | 96.79 (3/5) | 1.00x | 1.00x | 0.98x |
| **4.3 = 1a + 1b** | 0.130 (5/10) | 0.077 (9/10) | 0.126 (9/10) | 0.158 (5/10) | 0.155 (2/10) | 0.209 (5/10) | 96.79 (3/5) | 0.93x | 0.87x | 0.98x |
| no gate weights p | 0.136 (3/10) | 0.083 (5/10) | 0.135 (3/10) | 0.166 (2/10) | 0.171 (0/10) | 0.214 (4/10) | 96.63 (1/5) | 0.87x | 0.74x | 0.90x |
| one p per layer | 0.136 (1/10) | 0.107 (1/10) | 0.144 (2/10) | 0.194 (1/10) | 0.142 (7/10) | 0.208 (4/10) | 96.73 (2/5) | 1.00x | 1.00x | 0.91x |
| layers 2-3 gate on their own input | 0.140 (1/10) | 0.118 (0/10) | 0.158 (0/10) | 0.218 (0/10) | 0.190 (0/10) | 0.272 (1/10) | 96.23 (0/5) | 1.20x | 0.68x | 0.76x |
| gates shared by the 3 layers | 0.142 (1/10) | 0.117 (0/10) | 0.160 (2/10) | 0.180 (3/10) | 0.146 (7/10) | 0.213 (3/10) | 94.46 (0/5) | 0.82x | 0.66x | 0.67x |
| gates see a random 16-dim projection | 0.221 (0/10) | 0.132 (0/10) | 0.242 (0/10) | 0.280 (0/10) | 0.236 (0/10) | 0.274 (2/10) | 95.96 (0/5) | 1.26x | 0.52x | 0.63x |
| gates see a random 32-dim projection | 0.157 (0/10) | 0.099 (2/10) | 0.152 (1/10) | 0.207 (0/10) | 0.192 (1/10) | 0.246 (0/10) | 96.76 (2/5) | 1.85x | 0.68x | 0.75x |
| 3 hyperplanes (8 corners) | 0.140 (1/10) | 0.092 (1/10) | 0.152 (0/10) | 0.179 (2/10) | 0.169 (0/10) | 0.220 (3/10) | 96.09 (1/5) | 0.58x | 0.66x | 0.65x |
| 2 hyperplanes (4 corners) | 0.176 (0/10) | 0.111 (1/10) | 0.186 (0/10) | 0.224 (0/10) | 0.204 (0/10) | 0.271 (0/10) | 95.89 (0/5) | 0.33x | 0.41x | 0.42x |
| 3 hyperplanes, 24 units* | 0.135 (3/10) | 0.081 (4/10) | 0.147 (1/10) | 0.163 (3/10) | 0.161 (2/10) | 0.212 (4/10) | 96.39 (1/5) | 0.98x | 1.02x | 1.00x |
| keep 40% of inputs | 0.137 (3/10) | 0.095 (1/10) | 0.131 (5/10) | 0.181 (2/10) | 0.155 (2/10) | 0.218 (4/10) | 96.09 (2/5) | 0.80x | 0.89x | 0.84x |
| keep 30% of inputs | 0.148 (2/10) | 0.100 (0/10) | 0.130 (5/10) | 0.188 (1/10) | 0.159 (2/10) | 0.223 (3/10) | 95.99 (1/5) | 0.74x | 0.85x | 0.77x |
| 12 units per layer* | 0.130 (6/10) | 0.082 (4/10) | 0.131 (6/10) | 0.178 (2/10) | 0.162 (1/10) | 0.218 (3/10) | 96.43 (2/5) | 0.66x | 0.73x | 0.70x |
| 8 units per layer* | 0.133 (4/10) | 0.087 (5/10) | 0.132 (5/10) | 0.179 (1/10) | 0.165 (0/10) | 0.237 (1/10) | 96.06 (1/5) | 0.40x | 0.47x | 0.45x |
| 2 layers of 16* | 0.125 (9/10) | 0.082 (5/10) | 0.127 (7/10) | 0.185 (1/10) | 0.164 (1/10) | 0.239 (0/10) | 96.76 (2/5) | 0.63x | 0.72x | 0.73x |
| 2 layers of 24 | 0.121 (10/10) | 0.084 (4/10) | 0.125 (7/10) | 0.171 (4/10) | 0.156 (4/10) | 0.233 (2/10) | 96.96 (4/5) | 1.07x | 1.11x | 1.16x |
| 1 layer of 48 (no depth) | 0.136 (2/10) | 0.116 (0/10) | 0.149 (4/10) | 0.255 (0/10) | 0.183 (0/10) | 0.300 (0/10) | 96.59 (1/5) | 0.80x | 1.30x | 1.45x |
| gates updated every 2nd sample* | 0.147 (0/10) | 0.091 (2/10) | 0.151 (1/10) | 0.171 (1/10) | 0.193 (0/10) | 0.248 (0/10) | 96.43 (2/5) | 0.87x | 0.74x | 0.98x |

**The plain average across seed sets.** Each line is paired against 4.2 with the same seeds:

| Seed set | A pre | A post | B pre | B post | C pre | C post | Digits |
|---|---|---|---|---|---|---|---|
| 0 (table above) | -0.000 (5/10) | -0.005 (9/10) | -0.003 (9/10) | +0.000 (5/10) | +0.002 (2/10) | +0.002 (5/10) | +0.16 (3/5) |
| 1 | +0.001 (3/10) | -0.005 (9/10) | -0.003 (8/10) | +0.007 (3/10) | +0.002 (2/10) | -0.001 (6/10) | +0.14 (3/5) |
| 2 | -0.001 (8/10) | -0.005 (10/10) | -0.004 (8/10) | -0.010 (8/10) | +0.002 (1/10) | +0.005 (4/10) | -0.07 (1/5) |

- **Consistently better:** A after the drift (28 of 30 seeds) and B before it (25 of 30).
- **Consistently, very slightly worse:** C before the drift, by +0.002 NMSE, about 1%.
- **Within noise:** the other scores and digits.

On balance it is at least as accurate, and simpler.

## 4. What the study says about the model

- **The expensive parts earn their cost.** Four hyperplanes (16 corners), three layers of 16, gating every layer on the full input, per-unit gate weights `p` and a 60% connection budget each lose accuracy when cut. The losses are mostly after the drift, i.e. in how fast the model re-adapts. Every structural saving tried (fewer corners, units, layers or connections; shared or projected gates; less frequent gate updates) traded accuracy for cost at roughly the same rate.
- **The heatmap is soft.** At d = 64 the strongest corner carries 30% of a unit's weight on average, and 15 of 16 corners have h > 1e-3, so skipping weak corners would save little (measured on a d = 64 stream).
- **Depth matters for re-adaptation.** One wide layer of 48 units is much worse after the drift (B 0.255 vs 0.158).
- **If you need it cheaper and can lose a little accuracy,** the best trades measured are 12 units per layer (about 30% fewer operations and state; B and C post-drift worse by 0.01-0.02) and 3 hyperplanes (35-42% fewer operations; worse by 0.01-0.02 throughout). Neither is in the library, whose layer width and corner count are fixed in its SIMD layout.

## 4b. Cost/accuracy trade-off and pipelined layers (measured against 4.3)

Every row includes the 4.3 changes; the baseline row is 4.3 itself.

| Variant | A pre | A post | B pre | B post | C pre | C post | Digits 3 passes | Ops d=8 | Ops d=64 | State d=64 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4.3 (16 units) | 0.130 | 0.077 | 0.126 | 0.158 | 0.155 | 0.209 | 96.79 | 1.00x | 1.00x | 1.00x |
| 4.3, other seeds (noise) | 0.127 (4/10) | 0.081 (3/10) | 0.128 (6/10) | 0.163 (5/10) | 0.151 (8/10) | 0.213 (4/10) | 97.30 (4/5) | 1.00x | 1.00x | 1.00x |
| 4.3, other seeds (noise) | 0.124 (7/10) | 0.079 (3/10) | 0.126 (6/10) | 0.149 (8/10) | 0.150 (9/10) | 0.215 (3/10) | 97.06 (3/5) | 1.00x | 1.00x | 1.00x |
| pipelined layers | 0.190 (0/10) | 0.143 (0/10) | 0.199 (0/10) | 0.292 (0/10) | 0.190 (1/10) | 0.314 (0/10) | 95.09 (0/5) | 1.00x | 1.00x | 1.00x |
| 14 units | 0.130 (4/10) | 0.080 (1/10) | 0.127 (3/10) | 0.172 (2/10) | 0.158 (3/10) | 0.214 (4/10) | 96.76 (3/5) | 0.80x | 0.85x | 0.85x |
| **12 units** | 0.130 (5/10) | 0.082 (1/10) | 0.131 (3/10) | 0.178 (0/10) | 0.162 (1/10) | 0.218 (3/10) | 96.43 (2/5) | 0.66x | 0.72x | 0.71x |
| 10 units | 0.133 (3/10) | 0.087 (1/10) | 0.134 (4/10) | 0.177 (2/10) | 0.163 (1/10) | 0.225 (2/10) | 96.33 (1/5) | 0.52x | 0.59x | 0.58x |
| **8 units** | 0.133 (5/10) | 0.087 (1/10) | 0.132 (5/10) | 0.179 (1/10) | 0.165 (0/10) | 0.237 (2/10) | 96.06 (1/5) | 0.39x | 0.47x | 0.46x |
| 3 hyperplanes | 0.141 (1/10) | 0.085 (1/10) | 0.145 (0/10) | 0.174 (2/10) | 0.170 (1/10) | 0.221 (3/10) | 96.16 (1/5) | 0.56x | 0.65x | 0.65x |
| 12 units + 3 hyperplanes | 0.139 (0/10) | 0.089 (0/10) | 0.148 (1/10) | 0.188 (1/10) | 0.176 (0/10) | 0.238 (1/10) | 96.13 (1/5) | 0.38x | 0.47x | 0.47x |
| 12 units, keep 50% | 0.131 (5/10) | 0.086 (1/10) | 0.130 (3/10) | 0.178 (2/10) | 0.161 (1/10) | 0.225 (3/10) | 96.36 (1/5) | 0.61x | 0.69x | 0.66x |

**Reading the table.**
- At equal cost, fewer units per layer beat fewer hyperplanes: 8 units are about as cheap as 12 units with 3 hyperplanes, and more accurate on most scores.
- The library therefore offers the width as a setting, `Params::units` = 8 (the default since 5.0; 16 in 4.3), 12 or 16, rather than fewer corners.
- What narrowing costs is mostly re-adaptation after the drift (B and C post-drift). Before the drift and on digits, 12 units lose little.

**Pipelined layers (rejected).** Here layer 2 and layer 3 read the layer below's outputs from the previous sample. That would make all three rule stages independent, leaving the mixture as the only step that has to wait. It is much worse on every score (+0.03 to +0.13 NMSE, digits −1.7), so the chain between the layers within a sample is essential.

**Hybrid rule stages.** These keep part of the chain, so a sample needs 2 sequential rule stages instead of 3. All have the same cost as 4.3. NMSE change against 4.3, same seeds, with wins out of 10 (out of 5 for digits):

| Hybrid | A pre | A post | B pre | B post | C pre | C post | Digits |
|---|---|---|---|---|---|---|---|
| H1: L3 reads the previous sample's L2 | +0.010 (0) | +0.017 (0) | +0.016 (1) | +0.055 (0) | +0.010 (2) | +0.045 (0) | -0.60 (2) |
| H2: L2 reads the previous sample's L1 | +0.038 (0) | +0.039 (0) | +0.035 (0) | +0.051 (0) | +0.018 (3) | +0.054 (0) | -1.56 (0) |
| H4: L3 reads the mean of the previous L2 and the current L1 | +0.008 (0) | +0.016 (0) | +0.008 (2) | +0.047 (0) | +0.007 (1) | +0.035 (1) | -0.33 (1) |
| **H3, fan-out: L2 and L3 both read the current L1**, seed set 0 | -0.005 (9) | +0.004 (2) | -0.000 (6) | +0.020 (3) | +0.004 (0) | +0.024 (1) | +0.04 (2) |
| H3, seed set 1 | -0.007 (10) | +0.003 (1) | -0.002 (7) | +0.013 (2) | +0.003 (1) | +0.024 (0) | -0.24 (1) |
| H3, seed set 2 | -0.004 (10) | +0.002 (2) | +0.002 (3) | +0.023 (1) | +0.005 (2) | +0.020 (1) | +0.07 (3) |

- Any one-sample delay hurts.
- The fan-out (H3) has no delay and is the only usable hybrid. It is as good or slightly better before the drift and equal on digits, but re-adapts more slowly after the drift (+0.013 to +0.024 on B and C, every seed set).
- It saves no work. It shortens the critical path of a sample, which helps where the layers can run concurrently (the GPU). It is offered as the opt-in `Params::fanout`.

**In the library.** `Params::units` (8, 12 or 16) and `Params::fanout` are available on the CPU. Width 16 without fan-out is bit-identical to 4.3. Measured with `hgln_cli bench d nets N 1 units` (FORGE build, AVX-512 table, one thread, best of 3; bytes per network after the prune):

| Units | d=8, 1 net, f64 (µs) | d=8, f32 | d=64, 10 nets, f64 | d=64, 10 nets, f32 | Bytes d=8 (f64) | Bytes d=64 (f64) |
|---|---|---|---|---|---|---|
| 16 | 4.26 | 2.43 | 168 | 66.4 | 82,368 | 259,584 |
| 12 | 3.11 | 2.23 | 96.4 | 52.1 | 52,560 | 185,472 |
| 8 | 1.96 | 1.30 | 55.3 | 32.6 | 31,200 | 119,808 |

- The cuts are real: 12 units take 0.57-0.73x the time and 0.64-0.71x the memory; 8 units take 0.33-0.54x and 0.38-0.46x.
- In float32 on AVX-512, 12 units gain less (0.78-0.92x). A layer's 16 per-unit lanes fit one AVX-512 register, while 12 need an 8-lane and a 4-lane instruction.
- Computing all gates first and running the rule stages at the end gave no measurable CPU speed change (about ±3%), and neither did the fan-out: on one thread the work is the same. Their benefit is on the GPU, where they remove barriers.

## 5. Parallelism

Within one network, a sample has this dependency structure (exact, no change to the model). The counts below describe version 4.3 (3 x 16 = 48 units, chain). With the version 5.0 default it is 3 x 8 = 24 units (96 gates, a 24-term reduction), and with the fan-out layers 2 and 3 both read layer 1, so there are 2 sequential rule stages instead of 3.

1. **Gate pre-activations** of all three layers depend only on the input `u`, so all 192 gates can be computed in parallel, at once.
2. **Rule evaluation** is the only sequential part: layer 2's input is layer 1's output, so the rule stages run layer by layer (3 steps of 16 independent units each).
3. **The mixture** is a 48-term reduction.
4. **Updates** of all three layers are independent once the error signals are known, so all 48 units' updates can run in parallel, and within a unit the gate and rule updates are independent too.

So the critical path of a sample in 4.3 is 3 rule stages, 1 reduction and 1 update stage (2 rule stages with the 5.0 fan-out). The sequence comes from the chain between the layers, not from the global mixture. The mixture is a single 48-term reduction after the third stage, and the global error it produces feeds only the update. Computing all gates first and the three rule stages at the end is an exact reordering. Making the rule stages independent requires the pipelined variant, which is rejected above. Version 4.3 removes the renormalisation stage from it; on the GPU that is one block-wide barrier fewer per sample.

Across networks there is no dependency at all: class heads, ensemble members and streams parallelise perfectly. `run_streams`, `Classifier::train_batch` and the GPU `Bank` all exploit this.

Online learning over samples is inherently sequential: sample t+1 is predicted with the weights learned from sample t. No change to the model that keeps it an online learner can remove that dependency. Batching samples would make it a different algorithm.

## 6. Measured speed of 4.3 in the library

Gains are modest, because once vectorised the removed passes were cheap relative to the gate dot products and the rule updates. Single network, minimum over 40 blocks of 200 samples, `-O3 -march=native`, µs per sample:

| | 4.2 | 4.3 |
|---|---|---|
| d = 8, float64 | 6.2-6.4 | 6.2-6.3 |
| d = 64, float64 | 14.4-16.8 | 14.0-15.5 |
| d = 64, float32 | 6.8 | 5.5-6.1 (learn 3.4 to 2.6-2.9) |

State shrinks by the competence map: 768 values per network (6 KB in float64), about 2% at d = 64 and 7% at d = 8. In short, 4.3 gives a small, measurable improvement, not a large one. The large cuts in section 3 all cost accuracy.

## 7. Reproduce

```sh
cd research && python3 -I alg/gen_data.py          # writes research/alg/data (needs numpy, scikit-learn)
cd alg && g++ -O3 -march=native -std=c++17 -ffp-contract=off -pthread eval.cpp -o eval
./eval data baseline                                # about 7 s on 4 cores; one JSON line
./eval data mine uniform=1 norm_track=1 K=3 U=12    # any combination of the switches in hgln_var.hpp
python3 table.py results.jsonl                      # the table above from stored results
```

## 8. Version 5.0: cheaper structure plus three learning-rule additions, and an integer datapath

Version 5.0 starts from the cheapest structure section 4b found usable, **8 units per layer with the fan-out** (about 0.35x the work of 4.3), and wins back the accuracy it costs with three changes to how the gates learn, plus top-K corners. Version 5.0 is the library's default (`Params()`, the float `Net` and the integer `NetQ`, CPU and GPU); every change is also a separate setting, and `Params::v43()` gives version 4.3.

### 8a. The additions

- **Cross-layer latch** (`Params::latch = theta`, layers 2 and 3). Each gate bit of layers 2 and 3 becomes a set/reset flip-flop around the dead band `+-theta`, holding the same unit's bit of the layer it reads from (layer 1 with the fan-out) in between:
  ```
  st = sig(12 (z - theta)),  rt = sig(12 (-z - theta)),  q0 = the lower layer's bit
  a  = clip(st + (1 - st - rt) q0)            kap = d a / d(12 z) = st (1 - st)(1 - q0) + rt (1 - rt) q0
  ```
  It is stateless (it holds the *current* sample's lower bit, not a stored one), so it adds no state and no sequential dependency. The gate gradient becomes `12 kap (F1/a - F0/(1-a))`.
- **Latch credit** (`Params::latch_credit`). The part of the gradient that flows through the held bit, `(1 - st - rt)`, is routed to the lower layer's gate step (clipped like any gate step). Layers are updated 3, 2, 1, so layer 1 receives it in the same sample.
- **Gates learn from the global error** (`Params::gate_err = 2`), with `glob = 0.95`: the gate steps use `y - out` instead of each unit's mixed error. The rule steps still use the mixed error.
- **Top-K corners** (`Params::corners = 6`). Only the 6 largest of the 16 heatmap values are kept and renormalised for the rule mixture and the rule step; the gate gradient uses the soft values of those 6 corners and skips the other 10. A corner is selected when fewer than K corners have a larger value, ties going to the lower corner index. Saturated gates often give corners of exactly equal value, so this hard selection is sensitive to the last bit of the gate values. The library and the GPU kernels compute them with the same operations (`hgln_gate`, `hgln_exp`) and agree exactly. The NumPy reference uses NumPy's `exp`, can resolve such a tie the other way, and then differs by about 1e-6 (the float64 parity test allows 1e-4 for top-K modes). In float32 the rounding itself, about 1e-7, decides more of these near-ties, so a float32 network follows a different, equally valid trajectory from a float64 one (float32 against float64 C++ differs by up to about 0.2 over 2,000 samples). Float32 is therefore checked against the reference without top-K, and the float32 CPU and GPU agree exactly.

### 8b. Tuning and result

The settings were tuned by coordinate descent (`research/alg/tune.py`, log in `tune.log`, result in `tuned_v5.json`) on seed sets 0 and 1. The score is the mean relative NMSE change over the 6 synthetic windows and 3 time series (`narma10`, `mackey`, `regime`), minus 5% per digits point. Seed set 2 was never seen by the tuning. Against 4.3 (16 units, chain), NMSE change per window and digits points:

| Seed set | A pre | A post | B pre | B post | C pre | C post | narma | mackey | regime | Digits |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -7% | -6% | -3% | +8% | -6% | -7% | +2% | -13% | +3% | +0.54 |
| 1 | +0% | -9% | +5% | +8% | -7% | -8% | +3% | -13% | +3% | +0.43 |
| 2 (held out) | -2% | -10% | -1% | +19% | -4% | -12% | +3% | -12% | +1% | +0.50 |

Most scores improve and digits gain about half a point at about 0.35x the work. The cost is stream B after the drift (+8% to +19%): re-adaptation on that stream is the one place the narrow structure still shows.

### 8c. Integer datapath (`hgln::NetQ`, `include/hgln_qspec.hpp`)

`NetQ` runs the v5 model on int16 state and activations (12 to 16 bits, `Params::wbits` / `abits`), with fixed Q formats per array (spec in the header of `hgln_qspec.hpp`). The forward pass and the per-unit control scalars are integer arithmetic: a 1025-entry sigmoid table, int16 multiply-add dot products, integer divisions and square roots made exact by integer correction. The per-element learning steps (gate weights, hyperplanes and their renormalisation, rule weights) are **mixed**: float32 arithmetic on the int16 state, rounded back to the grid to nearest even (see below). It halves the model's memory against float32. The research reference (`hgln_var.hpp`, `intspec=1`) and the library agree bit for bit, and so does the GPU kernel.

**Rounding of the state updates.** Each update of `W`, `H` and `p` must be rounded back to its 16-bit grid. The first integer spec used stochastic rounding (Philox4x32-10), which is unbiased but cost most of the integer net's time. The study (`results46.jsonl`, float model with each array rounded to the grid after every update; 3 seed sets, mean of the 9 NMSE windows against 4.3, worst single window, digits points):

| Rounding of the 16-bit updates | Mean NMSE change | Worst window | Digits |
|---|---|---|---|
| none (v5 float) | -2.3% | +19% | +0.49 |
| stochastic | -1.8% | +20% | +0.35 |
| **round to nearest, ties to even** | **-2.4%** | **+15%** | **+0.30** |
| nearest + error feedback | -2.4% | +18% | +0.41 |
| floor (drop the low bits) | -4.0% | +18% | -1.21 |
| toward zero | -1.3% | +30% | -0.30 |

Round to nearest is as accurate as stochastic rounding: the updates are large relative to the grid, so the bias that stochastic rounding removes does not matter here. Plain truncation is worse (floor loses 1.2 digits points; toward zero gives a +30% worst window). **Version 5.0 uses round to nearest everywhere**: no random numbers are used after initialisation.

**The exact integer spec, re-run with round to nearest** (`results47.jsonl`: every row regenerated on the same data, including the 4.3 baselines and the previous stochastic spec):

| Integer spec | Mean NMSE change | Worst window | Digits (mean of 3 seed sets) |
|---|---|---|---|
| v5 float | -2.4% | +19% | 97.54% (+0.49) |
| int16, stochastic (previous) | -2.1% | +20% | 97.31% (+0.26) |
| **int16, round to nearest** | **-2.3%** | **+15%** | **97.39% (+0.34)** |
| int12, stochastic | -1.7% | +20% | 97.40% (+0.35) |
| int12, round to nearest | -1.9% | +16% | 97.13% (+0.08) |

The 4.3 baseline scores 97.05% on digits. Against the stochastic spec, same seeds, round to nearest changes the mean NMSE by -0.7% to +0.9% per seed set, which is noise.

```sh
B="uniform=1 norm_track=1 U=8 hybrid=3 theta=0.1 theta_mode=4 gate_err=2 credit=1 credit_scale=0.5 sparse_k=6 glob=0.95"
./eval data v5_float $B                                      # the float model with the v5 settings
./eval data v50_int16_nearest $B intspec=1 wbits=16 abits=16 # the exact integer spec (round to nearest)
./eval data v5_i16_nearest $B wbits=16 abits=16 round_mode=1 # float model, updates rounded to a 16-bit grid
# seed_off=1 / seed_off=2 for the other seed sets
```

**Mixed float / integer learning steps** (`results48.jsonl`). Done exactly in integers, the gate-weight and hyperplane steps and their renormalisation need 64-bit intermediates (up to 2^60) and an exact square root and division per hyperplane. That made learning the integer net's main cost. The spec now does these per-element steps in float32 instead, on the same int16 state: IEEE single precision, a fixed operation order, no fused multiply-adds, one correctly rounded square root and division per hyperplane, and the result rounded to the int16 grid to nearest even with `rnf(v) = (v + 1.5 2^23) - 1.5 2^23`. The forward pass and the per-unit control scalars stay integer. Float32 is deterministic under these rules, so every CPU target and GPU still gives the same bits. Every row regenerated on the same data, 3 seed sets:

| Integer spec | Mean NMSE change | Worst window | Digits (mean of 3 seed sets) |
|---|---|---|---|
| v5 float | -2.4% | +19% | 97.54% |
| int16, exact integer learning (previous) | -2.3% | +15% | 97.39% |
| **int16, mixed learning (version 5.0)** | **-2.2%** | **+15%** | **97.33%** |
| int12, exact integer learning | -1.9% | +16% | 97.13% |
| int12, mixed learning | -2.1% | +23% | 97.08% |

Against the exact spec, same seeds, the mixed int16 steps change the mean NMSE by -0.3%, -0.2% and +0.6% on the three seed sets, which is noise. At 12 bits one window is worse (+23% against +16%) while the mean is better. The speed gain is in `OPTIMISATION.md` section 5.

```sh
./eval data v50_int16_mixed $B intspec=1 wbits=16 abits=16   # the current spec (mixed learning steps)
```
