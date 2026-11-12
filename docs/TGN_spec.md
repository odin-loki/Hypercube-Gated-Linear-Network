# Hypercube Gated Linear Network (HGLN)

**Author:** Odin Loch
**Status:** Working spec (version 4.2). NumPy prototype, not peer reviewed. Supersedes versions 1 to 3. Version 4.1 sets the global-gradient share to 0.75 after tuning and validates on 5 fresh digit shuffles. Version 4.2 adds a unit-diversity measurement, an ablation of the competence weights, a verified C++ port, and a companion implementation spec (`TGN_impl_spec.md`).

## 1. Summary

A small online learner built from stacks of gated linear rules. Each unit places 4 learned hyperplanes, which carve its input space into 16 regions that are the corners of a 4-dimensional hypercube. An input produces a soft **heatmap over the 16 corners** (the product of four sigmoid gates, summing to 1). Each corner holds a linear rule, and the unit's output is the heatmap-weighted blend of the rules.

Version 4 adds a **competitive global function**: the model's prediction is a mixture over all hidden units, where each unit is weighted by its recent accuracy in the region the input falls in. Each unit also learns partly from its share of the mixture's error (a **global gradient**), so units cooperate instead of all chasing the target alone.

The model is stateless: output depends only on the current input. It learns online, predicting first and then learning from the true target.

**Main findings.**
- The units in a layer were almost perfectly redundant (mean correlation 0.88 to 0.98, effective rank about 1 of 16). A competitive mixture over all units, with no extra training, cut error sharply on every task.
- Sending part of each unit's error signal from the mixture's error (global gradient) improved results after a drift on all three synthetic families. A share of 0.75 was the best overall setting; 1.0 helped tasks A and C but hurt digits.
- Diversifying units by competence-weighted learning added almost nothing and was dropped for simplicity.
- A positive feedback loop (error raises a surprise-scaled step size, which raises error) was found and removed in version 3. Step size and connection budget are fixed.
- On 5 fresh digit shuffles the mixture beats online softmax regression in 4 of 5 shuffles after one pass and 5 of 5 after two and three passes, and beats the version 3 readout model in 5 of 5 at every pass.

## 2. Design principles

1. **Stateless model.** Time-based model state (recurrence, reservoirs, top-down loops) was tested and cut.
2. **Online learning.** Predict, then learn, one sample at a time. A consolidation step every 64 samples edits the connection mask.
3. **Matrix-based.** Rules, gates and masks are arrays; structure change is a mask edit.
4. **No error feedback into learning rates.** Step size and connection budget are fixed; the gate learning rate follows a fixed schedule.
5. **Hybrid of local and global learning.** Rules and gates are local to a unit. The global part is the competitive mixture and the shared share of its error.

## 3. Architecture

Input `x` (d values). Side information `u = [x; 1] / ||[x; 1]||` feeds every gate. Three hidden layers of Wd = 16 units (48 units total). Layer 1 takes `x`; later layers take the previous layer's 16 outputs. There is no separate readout layer: in version 3 the readout's output was never used once the mixture replaced it, and it feeds nothing, so it was removed with identical results.

For classification, one network is trained per class on +1 / -1 targets (one-vs-rest) and the class with the highest output wins.

Per layer (U units, n inputs):

| Array | Shape | Role |
|---|---|---|
| `H` | U x 4 x (d+1) | Unit-norm hyperplane normals (learned). |
| `p` | U x (d+1) | Learned per-input gate weights, clipped to [0, 3]. |
| `W` | U x 16 x (n+1) | Rule table: one linear rule per corner; last column is the +1 bias. |
| `M` | U x (n+1) | Connection mask. Starts all ones. |
| `m` | U x 16 | Competence map: running squared error of the unit's own prediction at each corner. |

**Forward (per unit).**
1. `z_b = H_b . (p * u)`, then gate `a_b = sigmoid(12 * z_b)` for the 4 hyperplanes.
2. Corner weight `h_c = prod_b ( a_b if bit b of c is 1, else 1 - a_b )`. The 16 weights are the heatmap and sum to 1.
3. Unit output `o = clip( sum_c h_c * (W_c . (input * M)), -2, 2 )`.

**Competitive global function.**
1. Competence of unit u at this input: `comp_u = 1 / ( sum_c h_c * m_c + 0.001 )`.
2. Responsibility `r_u = comp_u / sum_v comp_v` over all 48 units.
3. Prediction `y_hat = sum_u r_u * o_u`.

## 4. Online learning (per sample)

1. **Predict.** Forward through all layers, then the mixture above.
2. **Error signal per unit.** With the unit's own error `e_u = y - o_u` and the mixture's error `e_g = y - y_hat`:
   `es_u = 0.25 * e_u + 0.75 * e_g * r_u / sum_v r_v^2`.
   When all units have equal responsibility, the second term equals `e_g`, so the units move together to reduce the shared error.
3. **Rule update (convex, local).** Normalised LMS over all corners, weighted by the heatmap, using `es_u` and a fixed step `mu = 0.3`. This is a projection step.
4. **Gate update.** With rule outputs `f_c`, mix `m = sum h_c f_c`, `F1_b` and `F0_b` the sums of `h_c f_c` over corners with bit b equal to 1 and 0:
   `dm/dz_b = 12 * ( F1_b * (1 - a_b) - F0_b * a_b )`.
   Step `c_b = clip( glr * es_u * dm/dz_b, -0.05, 0.05 )`; `H_b += c_b * (p * u)`, then renormalise; `p += sum_b c_b * (H_b * u)`, clipped to [0, 3]. The gate learning rate follows `glr_t = glr1 + (glr0 - glr1) * exp(-t / tau)`.
5. **Competence update.** `m_c += 0.02 * h_c * ( e_u^2 - m_c )`, using the unit's own error.

**Consolidation (every 64 samples, after a 512-sample warmup).** Prune each unit's inputs to a fixed density of 0.6 by dropping the lowest `|W| x activity`. Layers with 4 or fewer inputs stay dense. With a fixed budget this is a one-way dense-to-sparse step.

## 5. Default hyperparameters

| Name | Value |
|---|---|
| Hidden layers x units | 3 x 16 (48 units), no readout |
| Hyperplanes per unit (cube dimension) | 4 (16 corners) |
| Gate sharpness | 12 |
| Rule step size `mu` | 0.3 |
| Global-gradient share | 0.75 (tuned over 0.5, 0.75, 1.0) |
| Competence rate, initial `m` | 0.02, 0.3 |
| Connection density after warmup | 0.6 |
| Warmup, consolidation period | 512, 64 |
| Gate learning rate, 64-input classification | fixed 0.02 |
| Gate learning rate, 8-input streams | 0.2 annealed to 0.05, tau 1000 |

## 6. Results

Normalised MSE (0 is perfect, 1 means no better than predicting the mean). Online, single pass. 8 inputs, 4096 samples, function changes at sample 2048. 10 seeds, the same seeds for every row. Each cell is **pre-drift (late) -> post-drift (late)**.

| Model | A: sin/cos products | B: random projections | C: radial bumps |
|---|---|---|---|
| Previous baseline (top-2 codes, adaptive step) | .256 -> .184 | .407 -> .438 | .288 -> .360 |
| Version 3 (readout output, annealed gate rate) | .180 -> .146 | .198 -> .248 | .236 -> .345 |
| Competitive mixture only | .124 -> .117 | .141 -> .231 | .181 -> .290 |
| Version 4, share 0.5 | .125 -> .097 | .133 -> .191 | .162 -> .242 |
| **Version 4.1, share 0.75 (default)** | **.125 -> .086** | **.132 -> .172** | **.149 -> .213** |
| Version 4, share 1.0 | .117 -> .071 | .112 -> .185 | .123 -> .185 |
| Mixture + diversify (dropped) | .125 -> .118 | .140 -> .223 | .179 -> .284 |
| Mixture + diversify + global gradient (dropped) | .127 -> .098 | .133 -> .185 | .161 -> .238 |

Against the competitive mixture alone, the global gradient wins in 10/10 seeds on A late and in 10/10 seeds on all three scores for task C. On B it wins 7/9/7 of 10. Diversifying (competence-weighted learning with a conscience term) changed results by 0.008 or less and was dropped.

**Redundancy of the units** (task A, after 1,536 samples, 500 probe inputs; measured before the global gradient was added):

| Layer | Mean abs. correlation between units | Effective rank (of 16) | Single-unit NMSE (mean / best) | Mean-of-units NMSE |
|---|---|---|---|---|
| 1 | 0.88 | 1.2 | .252 / .156 | .163 |
| 2 | 0.97 | 1.1 | .153 / .130 | .123 |
| 3 | 0.98 | 1.0 | .188 / .157 | .156 |

The old readout scored .232 on the same probe, worse than simply averaging layer-2 units. Whether the units are more diverse after the global gradient has not been re-measured.

**Digits** (one-vs-rest, 1,198 stream / 599 test, gate lr 0.02, holdout % after 1 / 2 / 3 passes):

| Shuffle | Model | 1 | 2 | 3 |
|---|---|---|---|---|
| 10 | Version 3 readout | 90.5 | 95.5 | 96.8 |
| 10 | Competitive mixture | 95.5 | 97.2 | 98.5 |
| 10 | **Version 4 (mixture + global gradient)** | 95.5 | 98.2 | 98.8 |
| 10 | Online softmax regression | 93.0 | 94.7 | 96.5 |
| 11 | Version 3 readout | 87.8 | 92.8 | 95.2 |
| 11 | Competitive mixture | 93.2 | 95.0 | 96.3 |
| 11 | Online softmax regression | 89.0 | 89.8 | 91.0 |

Version 4 was tested on one shuffle only. Across 5 fresh shuffles the version 3 readout model reached 96.7% after 5 passes versus 94.2% for softmax and 92.8% for a 64x64 per-sample MLP. Offline models measured on different shuffles reach about 97% (random forest, gradient boosting, logistic regression, MLP) and 99% (kNN), so these figures are not a like-for-like comparison.

**Global-gradient share** (10 seeds, same as above; wins are against share 0.5, pre-drift late / post-drift early / post-drift late):

| Share | A | B | C |
|---|---|---|---|
| 0.5 | .125 -> .097 | .133 -> .191 | .162 -> .242 |
| 0.75 | .125 -> .086 (5/6/10) | .132 -> .172 (7/7/8) | .149 -> .213 (10/10/10) |
| 1.0 | .117 -> .071 (9/8/10) | .112 -> .185 (10/5/4) | .123 -> .185 (10/7/9) |

Share 1.0 is best on A and C but loses to 0.75 on B post-drift and on digits (below), so 0.75 is the default.

**Digits, 5 fresh shuffles** (10-14; 1,198 stream / 599 test; gate lr 0.02; mean holdout % after 1 / 2 / 3 passes; wins are shuffles out of 5 against the row named):

| Model | 1 | 2 | 3 | Beats softmax | Beats v3 readout |
|---|---|---|---|---|---|
| Online softmax regression | 90.8 | 92.7 | 93.7 | - | - |
| Version 3 readout | 88.9 | 92.2 | 94.5 | - | - |
| Version 4, share 0.5 | 93.2 | 95.5 | 96.5 | 4 / 5 / 5 | 5 / 5 / 5 |
| **Version 4.1, share 0.75** | 93.1 | 95.4 | **97.0** | 4 / 5 / 5 | 5 / 5 / 5 |
| Version 4, share 1.0 | 92.4 | 94.0 | 95.2 | 3 / 3 / 5 | 5 / 5 / 3 |

Shuffles 10 and 11 were also used for the earlier mixture and diversify checks, so they are not fully fresh for this model family.

## 7. What did not work

| Attempt | Outcome |
|---|---|
| Surprise-scaled step size | Positive feedback loop and a 15-25 point dip at width 64. Removed. |
| Diversify units by competence-weighted learning | No meaningful gain. Dropped. |
| Unsupervised router, learned co-activation graph, free-code lattice with smoothing | Worse than gradient-trained gates; the lattice collapsed to roughly a linear model. |
| Load balancing of hyperplane offsets, rule smoothing across cube edges | Worse. No collapse occurs when corners come from 4 hyperplanes. |
| Purely geometric gates (fixed random hyperplanes) | Worse in 8 dimensions. |
| Heatmap passed up as features (4 gate bits per unit) | Worse on digits. |
| Global linear rule added to each unit | No clear gain. |
| Reservoir state, top-down feedback, competing networks | Little or no gain. Cut by design. |

## 7b. Version 4.2 findings

- **Competence weighting adds almost nothing.** Replacing the competitive weights `r_u` with uniform weights (global-gradient share still 0.75) gives window-mean prequential NMSE (windows 1536-2048 / 3584-4096) of 0.125 / 0.082 on task A against 0.125 / 0.086 competitive, 0.129 / 0.166 on B against 0.132 / 0.172, and 0.151 / 0.212 on C against 0.149 / 0.213. All differences are within seed noise. The mixture behaves as a plain average of units trained with a shared global gradient. The competence map `m` and the `comp` computation could be dropped to simplify the port; this has been shown on these three tasks only, not on digits, so the reference keeps them for now.
- **Unit diversity falls as the global-gradient share rises.** With share 0 the mean effective number of units is 44.8 of 48; at 0.75 it is 45.2; at 1.0 it is 40.8. Layer-1 mean absolute pairwise correlation between unit outputs goes from 0.87 (share 0) to 0.78 (0.75) to 0.27 (1.0), the participation-ratio rank of the unit outputs rises from 1.25 to 1.52 to 4.8, and mean individual unit NMSE rises from 0.28 to 0.39 to 1.52 (layers 2 and 3 show the same pattern). At the default 0.75 the units stay individually useful; at 1.0 they become partial terms that only work in sum.

**Simplifications in the port.** Several state arrays in the research code are dead in version 4.1 (gate codes, use counts, balancing offsets, recent-input and error buffers). The structure step only prunes, once, at sample 512, because each unit then already holds its target number of inputs. The C++ port omits all of this and matches the NumPy reference to about 1e-15 (float64) and 1e-6 (float32).

**C++ speed.** The first, unoptimised port took about 11 to 14 us per sample for one network at d = 8, and about 465 to 490 us (float64) and 310 to 330 us (float32) per sample for 10 networks at d = 64. That was roughly 6 to 7 times slower than a matched 3x16 MLP at d = 8 and 17 to 22 times slower at d = 64. The optimised library takes about 6 us at d = 8, and at d = 64 with 10 networks about 246 us (float64) and 72 us (float32) on one thread, or 52 us and 23 us with the networks on 4 threads. The MLP takes about 2 us and 26 us. Details in `TGN_impl_spec.md` section 7 and `OPTIMISATION.md`.

## 7c. Version 4.3

Two changes to the procedure, chosen after measuring 23 variants on the benchmarks above (details: `ALGORITHM.md`):
1. **Exact norm tracking.** The hyperplanes are stored unnormalised, with their squared lengths updated in closed form. This is the same model, with no renormalisation pass.
2. **Plain average.** The mixture is a plain average and the competence map is gone: `es_i = (1 - glob)(y - o_i) + glob (y - out)`. Over 3 seed sets and 5 digit shuffles it is at least as accurate. It is better on A post-drift (28/30 seeds) and B pre-drift (25/30), and 0.002 worse on C pre-drift.

Cutting corners, units, layers, connections or gate parameters, or sharing or projecting the gates, all lost accuracy.

## 8. Limitations

- **Selection.** The mixture and the global gradient were chosen after seeing results on the same synthetic seeds and digits shuffles used for comparison. Numbers are mildly optimistic.
- **Partly tuned.** The global share was tuned over three values. The competence rate and gate rates were not tuned for version 4.
- **Digits.** Test sets are 599 samples, so differences of about 1 point are noise. The share was tuned on the same synthetic seeds and digit shuffles used to report it.
- **Gate learning rate is task-dependent.** Low (0.02) suited 64-input digits; higher, annealed values suited the 8-input streams. A rule of about 1.6 / d fits both but is untested.
- **Speed.** Measured in C++ (section 7b): still slower than a matched small MLP on one thread (about 3x at d = 8), though SIMD-tuned and multithreaded across networks. Not sparse or event-driven. The CUDA kernel compiles to PTX and is verified on a CPU emulator; it has since run on an NVIDIA RTX 3090, bit-identical to the CPU library (GPU speed not studied).
- **Competence weighting** is not shown to help (section 7b), and was tested on the three synthetic tasks only.
- **Baselines.** Online softmax regression and per-sample MLPs. Stronger tuning could change the comparison.
- **Width.** Width 16 mainly.
- **Dense-sparse-dense cycle and the surprise signal** are no longer part of the model.

## 9. Next steps

1. Decide whether to drop the competence map (section 7b), after checking digits. Tune the competence rate only if it is kept.
2. More real datasets with stronger baselines, and more passes on digits.
3. Check the 1.6 / d rule for the gate rate on other input sizes.
4. Wider networks (64-128).
5. Keep only the strongest few corners active, then measure the accuracy cost.
6. Run the CUDA kernel on real GPUs (for example through FORGE's driver-API runtime, `FORGE.md`). The SIMD and multithreaded C++ library is done. *Done for NVIDIA (RTX 3090, bit-identical through FORGE's runtime); AMD is still open.*

## 10. Reference implementation

Single file, verified against the research code on a fixed seed (largest difference 2e-16, share 0.75).

```python
net = HGLN(seed=0, d=8)                              # 8-input stream: gate lr 0.2 annealed to 0.05
# net = HGLN(seed=0, d=64, glr0=0.02, glr1=0.02)     # 64-input classification: fixed 0.02
for x, y in stream:
    y_hat = net.predict(x)   # inference
    net.learn(y)             # learn from the true target
```

```python
"""Hypercube Gated Linear Network (HGLN), version 4.
Stateless online learner. Each unit has 4 learned hyperplanes -> a 16-corner heatmap -> a blend of 16 local linear rules.
The prediction is a competitive mixture over all hidden units (weights = each unit's local competence), and each unit
learns from a blend of its own error and its responsibility-share of the mixture's error (global gradient).
Fixed step size and connection budget (no error feedback into learning rates); gate learning rate on a fixed schedule.
Author: Odin Loch"""
import numpy as np
BETA = 12.0

def feat(x):
    v = np.append(x, 1.0)
    return v / np.linalg.norm(v)

class GatedLayer:
    """U units; each unit owns K gated linear rules (rows of W) and K gate codes (rows of E)."""
    def __init__(s, U, n_in, K, d, rng, glr):
        s.U, s.n, s.K, s.d, s.rng, s.glr = U, n_in, K, d, rng, glr
        p = rng.uniform(-1, 1, (U, K, d))
        v = np.concatenate([p, np.ones((U, K, 1))], 2)
        s.E = v / np.linalg.norm(v, axis=2, keepdims=True)      # gate codes (U,K,d+1)
        s.p = np.ones((U, d + 1))                                # learned per-input gate weights
        s.W = np.zeros((U, K, n_in + 1))                         # rule table; last column = bias (+1)
        s.M = np.ones((U, n_in + 1))                             # connection mask (dense at start)
        s.alive = np.ones((U, K), bool)                          # which codes exist
        s.use = np.full((U, K), 1.0 / K)                         # code usage (EMA)
        s.G = np.zeros((U, n_in + 1))                            # accumulated |error * input| (regrowth signal)
        s.A = np.full((U, n_in + 1), 1e-3)                       # input activity (EMA)
        s.bu = np.zeros((64, d + 1)); s.be = np.zeros((64, U)); s.bi = 0   # recent side-info / errors
        s.ar = np.arange(U)

    def forward(s, a, u):
        s.inp = np.append(a, 1.0); s.u = u; s.pu = s.p * u[None]
        sc = np.where(s.alive, np.einsum('ukd,ud->uk', s.E, s.pu), -1e9)
        s.idx = np.argpartition(-sc, 1, axis=1)[:, :2]           # top-2 codes per unit (sparse routing)
        top = np.take_along_axis(sc, s.idx, 1)
        z = np.exp(BETA * (top - top.max(1, keepdims=True)))
        s.sw = z / z.sum(1, keepdims=True)
        s.xm = s.inp[None] * s.M
        s.f = np.einsum('ukn,un->uk', s.W[s.ar[:, None], s.idx], s.xm)
        s.pred = np.clip((s.sw * s.f).sum(1), -2, 2)
        return s.pred

    def update(s, y, mu):
        e = y - s.pred
        if s.glr > 0:   # error-driven gate: gradient of the unit's own loss through the 2-way routing
            c = np.clip(s.glr * e * (s.f[:, 0] - s.f[:, 1]) * BETA * s.sw[:, 0] * s.sw[:, 1], -0.05, 0.05)
            j0, j1 = s.idx[:, 0], s.idx[:, 1]
            E0, E1 = s.E[s.ar, j0], s.E[s.ar, j1]
            s.p = np.clip(s.p + c[:, None] * ((E0 - E1) * s.u[None]), 0.0, 3.0)
            n0, n1 = E0 + c[:, None] * s.pu, E1 - c[:, None] * s.pu
            s.E[s.ar, j0] = n0 / np.linalg.norm(n0, axis=1, keepdims=True)
            s.E[s.ar, j1] = n1 / np.linalg.norm(n1, axis=1, keepdims=True)
        # convex local step on the two active rules (normalised LMS), no backprop
        c = mu * e / (1e-3 + (s.sw ** 2).sum(1) * (s.xm ** 2).sum(1))
        s.use *= 0.99
        for k in range(2):
            j = s.idx[:, k]
            s.W[s.ar, j] += (c * s.sw[:, k])[:, None] * s.xm
            s.use[s.ar, j] += 0.01 * s.sw[:, k]
        s.G = 0.98 * s.G + 0.02 * np.abs(e)[:, None] * np.abs(s.inp)[None]
        s.A = 0.98 * s.A + 0.02 * np.abs(s.inp)[None]
        s.bu[s.bi] = s.u; s.be[s.bi] = np.abs(e); s.bi = (s.bi + 1) % 64

    def structure_step(s, g):
        """Slow timescale. Surprise g sets the connection budget: calm -> prune, surprise -> regrow."""
        rho = float(np.clip(0.6 * g, 0.4, 1.0)); n = s.n
        nt = n if n <= 4 else int(round(rho * n))                # tiny input layers stay dense
        for u in range(s.U):
            act = np.where(s.M[u, :n] > 0)[0]; ina = np.where(s.M[u, :n] == 0)[0]
            if len(act) > nt:                                    # prune lowest |weight| x activity
                keep = np.abs(s.W[u][:, :n]).mean(0) * s.A[u, :n]
                s.M[u, act[np.argsort(keep[act])[:len(act) - nt]]] = 0
            elif len(act) < nt and len(ina):                     # regrow where error x input is largest
                gr = ina[np.argsort(-s.G[u, :n][ina])[:nt - len(act)]]
                s.M[u, gr] = 1; s.W[u][:, gr] = 0
            al = np.where(s.alive[u])[0]
            if g < 1.15 and len(al) > 4:                         # calm: drop a rarely used code
                j = al[np.argmin(s.use[u, al])]
                if s.use[u, j] < 0.35 / len(al): s.alive[u, j] = False
            elif g > 1.3:                                        # surprised: respawn a code where error was worst
                dead = np.where(~s.alive[u])[0]
                if len(dead):
                    uv = s.bu[np.argmax(s.be[:, u])]
                    w = al[np.argmax(s.E[u, al] @ (s.p[u] * uv))]
                    j = dead[0]; s.E[u, j] = uv / np.linalg.norm(uv)
                    s.W[u, j] = s.W[u, w]; s.alive[u, j] = True; s.use[u, j] = 1.0 / len(al)


class HC(GatedLayer):
    """Hypercube from 4 learned hyperplanes: corner c (4 bits) gets weight prod_b sigma(+-beta*z_b).
    The 16 corner weights are the heatmap over the tesseract (they sum to 1). Neighbours = one bit flipped.
    MoE-style load balancing: each hyperplane's offset is nudged so it splits the data about 50/50."""
    def __init__(s,U,n_in,K,d,rng,glr,balance=True,lam=0.0,gam=0.02):
        super().__init__(U,n_in,16,d,rng,glr)
        h=rng.normal(size=(U,4,d+1)); h[:,:,-1]=0.0
        s.H=h/np.linalg.norm(h,axis=2,keepdims=True)
        s.B=((np.arange(16)[:,None]>>np.arange(4)[None])&1).astype(float)
        s.delta=np.zeros((U,4)); s.mb=np.full((U,4),0.5); s.balance,s.lam,s.gam=balance,lam,gam
    def forward(s,a,u):
        s.inp=np.append(a,1.0); s.u=u; s.pu=s.p*u[None]
        z=np.einsum('ubd,ud->ub',s.H,s.pu)+s.delta
        s.a=np.clip(1/(1+np.exp(-BETA*z)),1e-4,1-1e-4)
        s.h=np.prod(np.where(s.B[None]>0.5,s.a[:,None,:],1-s.a[:,None,:]),axis=2)
        s.xm=s.inp[None]*s.M
        s.f=np.einsum('ucn,un->uc',s.W,s.xm)
        s.mix=(s.h*s.f).sum(1); s.pred=np.clip(s.mix,-2,2); return s.pred
    def update(s,y,mu,gscale=1.0):
        e=y-s.pred
        if s.glr>0:
            hf=s.h*s.f; F1=hf@s.B; F0=hf.sum(1,keepdims=True)-F1
            g=BETA*(F1*(1-s.a)-F0*s.a)                      # d mix / d z_b
            c=np.clip(s.glr*gscale*e[:,None]*g,-0.05,0.05)
            dp=np.einsum('ub,ubd->ud',c,s.H*s.u[None,None,:])
            s.H=s.H+c[:,:,None]*s.pu[:,None,:]
            s.H/=np.linalg.norm(s.H,axis=2,keepdims=True)
            s.p=np.clip(s.p+dp,0.0,3.0)
        s.mb=0.99*s.mb+0.01*s.a
        if s.balance: s.delta-=s.gam*(s.mb-0.5)               # load balancing, no auxiliary loss
        c2=mu*e/(1e-3+(s.h**2).sum(1)*(s.xm**2).sum(1))
        s.W+=(c2[:,None]*s.h)[:,:,None]*s.xm[:,None,:]
        s.use=0.99*s.use+0.01*s.h
        s.G=0.98*s.G+0.02*np.abs(e)[:,None]*np.abs(s.inp)[None]
        s.A=0.98*s.A+0.02*np.abs(s.inp)[None]
        s.bu[s.bi]=s.u; s.be[s.bi]=np.abs(e); s.bi=(s.bi+1)%64
    def structure_step(s,g):
        super().structure_step(g); s.alive[:]=True
        if s.lam>0:                                           # light smoothing of RULES across cube edges
            Wn=np.mean([s.W[:,np.arange(16)^(1<<b)] for b in range(4)],axis=0)
            s.W=(1-s.lam)*s.W+s.lam*Wn


class HC2(HC):
    """Hypercube unit with a per-corner competence map m (running squared error of the unit's own prediction)."""
    def __init__(s, U, n_in, K, d, rng, glr):
        super().__init__(U, n_in, K, d, rng, glr, False, 0.0); s.m = np.full((U, 16), 0.3)
    def comp(s):                                           # input-dependent competence of each unit
        return 1.0 / ((s.h * s.m).sum(1) + 1e-3)
    def update(s, y, mu, es):                              # es: the error signal this unit learns from
        e = y - s.pred
        s.m = s.m + 0.02 * s.h * ((e ** 2)[:, None] - s.m)
        hf = s.h * s.f; F1 = hf @ s.B; F0 = hf.sum(1, keepdims=True) - F1
        g = BETA * (F1 * (1 - s.a) - F0 * s.a)             # d mix / d z_b
        c = np.clip(s.glr * es[:, None] * g, -0.05, 0.05)
        dp = np.einsum('ub,ubd->ud', c, s.H * s.u[None, None, :])
        s.H = s.H + c[:, :, None] * s.pu[:, None, :]; s.H /= np.linalg.norm(s.H, axis=2, keepdims=True)
        s.p = np.clip(s.p + dp, 0.0, 3.0)
        c2 = mu * es / (1e-3 + (s.h ** 2).sum(1) * (s.xm ** 2).sum(1))
        s.W += (c2[:, None] * s.h)[:, :, None] * s.xm[:, None, :]
        s.use = 0.99 * s.use + 0.01 * s.h
        s.G = 0.98 * s.G + 0.02 * np.abs(es)[:, None] * np.abs(s.inp)[None]
        s.A = 0.98 * s.A + 0.02 * np.abs(s.inp)[None]
        s.bu[s.bi] = s.u; s.be[s.bi] = np.abs(e); s.bi = (s.bi + 1) % 64

class HGLN:
    """3 hidden layers of Wd units (no separate readout). Prediction = competitive mixture over all 3*Wd units."""
    def __init__(s, seed, d, Wd=16, glr0=0.2, glr1=0.05, tau=1000.0, mu=0.3, glob=0.75, warm=512):
        r = np.random.default_rng(seed)
        s.L = [HC2(Wd, d, 16, d, r, glr0), HC2(Wd, Wd, 16, d, r, glr0), HC2(Wd, Wd, 16, d, r, glr0)]
        s.Wd, s.glr0, s.glr1, s.tau, s.mu, s.glob, s.warm, s.t = Wd, glr0, glr1, tau, mu, glob, warm, 0
    def predict(s, x):
        x = np.asarray(x, float); u = feat(x); a = x
        for L in s.L: a = L.forward(a, u)
        s.o = np.concatenate([L.pred for L in s.L])
        comp = np.concatenate([L.comp() for L in s.L])
        s.r = comp / comp.sum()                            # competitive weights over units
        s.out = float((s.r * s.o).sum()); return s.out
    def learn(s, y):
        gl = s.glr1 + (s.glr0 - s.glr1) * np.exp(-s.t / s.tau)          # fixed gate-rate schedule
        for L in s.L: L.glr = gl
        e_loc = y - s.o; e_glob = y - s.out
        es = (1 - s.glob) * e_loc + s.glob * e_glob * s.r / (s.r ** 2).sum()   # own error + responsibility-share of mixture error
        for i, L in enumerate(s.L): L.update(y, s.mu, es[i * s.Wd:(i + 1) * s.Wd])
        s.t += 1
        if s.t >= s.warm and s.t % 64 == 0:                              # consolidation: fixed density 0.6
            for L in s.L: L.structure_step(1.0)
```
