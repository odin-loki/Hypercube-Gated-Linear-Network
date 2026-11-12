# Algorithm study (version 4.3)

`hgln_var.hpp` is a scalar, readable HGLN with switches for each candidate change to the algorithm. With no switches it is version 4.2, and it matches the library to about 4e-15.

- `eval.cpp` scores one variant on the spec's benchmarks: synthetic streams A, B and C (10 seeds) and digits (5 shuffles). It runs all jobs in parallel.
- `gen_data.py` writes the benchmark data.
- `results.jsonl` holds the stored results, and `table.py` / `summ.py` summarise them.

Findings and method: [docs/ALGORITHM.md](../../docs/ALGORITHM.md).

## Version 5.0 integer specification (`intspec=1`)

`QNet` in `hgln_var.hpp` runs the exact integer specification of `include/hgln_qspec.hpp` (Philox4x32-10 stochastic
rounding, sigmoid table, top-k corners, integer divisions and square roots). It is the reference the library's
`hgln::NetQ` must match bit for bit (`tests/test_q.cpp`). The v5 rows in `results45.jsonl`:

```sh
B="uniform=1 norm_track=1 U=8 hybrid=3 theta=0.1 theta_mode=4 gate_err=2 credit=1 credit_scale=0.5 sparse_k=6 glob=0.95"
./eval data v5_float $B                                   # float model with the v5 settings
./eval data v50_int16 $B intspec=1 wbits=16 abits=16      # the integer specification (also wbits=abits=12)
./eval data v50_int16_ginit $B intspec=1 intspec_ginit=1  # same, but from the float model's initial hyperplanes
# add seed_off=1 / seed_off=2 (rows *_s1, *_s2) for the other seed sets
```
