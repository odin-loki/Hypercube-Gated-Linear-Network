# Algorithm study (version 4.3)

`hgln_var.hpp` is a scalar, readable HGLN with switches for each candidate change to the algorithm. With no switches it is version 4.2, and it matches the library to about 4e-15.

- `eval.cpp` scores one variant on the spec's benchmarks: synthetic streams A, B and C (10 seeds) and digits (5 shuffles). It runs all jobs in parallel.
- `gen_data.py` writes the benchmark data.
- `results.jsonl` holds the stored results, and `table.py` / `summ.py` summarise them.

Findings and method: [docs/ALGORITHM.md](../../docs/ALGORITHM.md).
