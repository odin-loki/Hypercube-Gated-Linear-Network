# Research code (NumPy prototypes and stored results)

These are the experiment scripts behind the numbers in `docs/TGN_spec.md`, kept for the record. They are not a clean API.
`python/hgln_ref.py` is the verified single-file reference; use that for anything new.

- Chain of prototypes: `tgn_final.py` -> `tgn_v5.py` -> `tgn_v8.py` -> `tgn_v10.py` -> `tgn_v11.py` (`NetT`, the version in the spec).
- `t7.py` and `t24.py` define the synthetic streams A, B, C (8 inputs, 4096 samples, drift at 2048).
- `t26.py` share sweep and digits; `t28.py` unit diversity; `t29.py` competence-weight ablation.
- `out*.txt` are the stored JSON results (one line per run).
- `hgln_v1_single_file.cpp` is the first single-file C++ port, superseded by `include/hgln.hpp`.

Run from this folder, for example `python3 t29.py`. Several scripts take minutes to hours on one CPU core. They were not re-run
when the package was assembled; the stored outputs are what the spec reports.
