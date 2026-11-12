# Hypercube Gated Linear Network (HGLN): a stateless online-learning algorithm

**From:** Odin Loch, independent researcher, Sydney, Australia
**Contact:** odin.loch@outlook.com.au | GitHub: odin-loki

## What it is

HGLN is a neural-style learner that trains and predicts at the same time, one sample at a time. It has no batches, no stored training data and no memory of the past beyond its current weights. It keeps adapting when the data changes, which suits streaming sensor and signal work.

## How it works

- **Gating.** Each of 48 units (3 layers of 16) learns 4 hyperplanes. These split the input space into 16 regions, and the unit blends 16 simple linear rules by how strongly the input falls in each region.
- **Global mixture.** The output is a weighted average over all 48 units. Each unit trains partly on its own error and partly on its share of the combined error (75% by default), so the units cooperate.
- **Fixed budget.** After 512 samples each unit keeps a fixed 60% of its inputs.
- **Cost.** There is no backpropagation through the network and no step size that reacts to error. Each update touches only local state.

## Results so far

From a NumPy prototype on synthetic drifting streams (10 seeds) and the public scikit-learn digits set (5 shuffles).

- It recovers after a mid-stream change in the target. On three synthetic tasks the normalised error (lower is better) fell from about 0.15-0.35 to 0.09-0.21 between the windows just before and just after the late part of the stream.
- On digits, one-vs-rest, it reaches about 97.0% holdout accuracy on average after 3 passes, and beat online softmax regression in all 5 shuffles at 2 and 3 passes. Offline methods reach about 97-99% on this set, so no claim is made of exceeding them.

## Status and limits

- **Implementation.** An optimised C++ library (SIMD, multithreaded) matches the NumPy reference to about 1e-15 (float64) and 1e-6 (float32). Built on the author's FORGE runtime, it picks the best CPU vector instructions at run time. The GPU version targets NVIDIA and AMD through FORGE and matches the CPU results exactly in emulation, but has not been run on a GPU. Licensed under the GNU AGPL v3.0 or later.
- **Speed.** About 6 microseconds per sample at 8 inputs on one CPU thread, about 3 times slower than a matched small MLP (the first version was 6-7 times slower). Independent networks, such as class heads, run in parallel across cores: 10 heads at 64 inputs take about 23 microseconds per sample in float32 on 4 cores.
- **Component test.** An ablation showed the competence weighting adds almost nothing beyond averaging. The gain comes from the shared global error signal.
- **Evidence.** Tuning and comparison used the same seeds and shuffles, so the figures are mildly optimistic. Only small synthetic and digits tasks have been tested, against basic baselines. Nothing has been tested on real defence signals, at larger width, or against stronger online learners.

## Where it could help

Streaming classification and regression where the data drifts and compute or memory is limited, such as RF and sensor processing at the edge. This is an application hypothesis, not a result.

## Next steps

Test the GPU version on real hardware, run wider networks (64-128), and run stronger baselines on real datasets.

Full specification and reference code are available on request.
