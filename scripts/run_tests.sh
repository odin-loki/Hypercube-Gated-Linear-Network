#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Runs every test that can run on this machine. Needs: g++ (C++20), python3 with numpy and scikit-learn.
set -e
cd "$(dirname "$0")/.."
echo "== C++ behavioural tests"; ./build/test_api
echo "== NumPy vs C++ parity (versions 4.3 and 4.2; float64 and float32; d=8 and d=64; 8, 12 and 16 units per layer; fan-out)"
for u in 8 12 16; do for m in v43 v42; do for d in 8 64; do
  python3 python/parity.py build/hgln_d $d 1e-12 $m $u; python3 python/parity.py build/hgln_f $d 1e-4 $m $u
done; done; done
for d in 8 64; do python3 python/parity.py build/hgln_d $d 1e-12 v43f 16; python3 python/parity.py build/hgln_f $d 1e-4 v43f 16; done
echo "== CUDA kernel logic on the CPU emulator vs the C++ library (this does NOT run on a GPU)"
# args: d nets T threads shared competence [far]; competence 0 = 4.3 plain average, 1 = 4.2 competence mixture;
# far = 1 starts two layers at the edge of the norm range (conditional renormalisation); T > 4096 crosses the periodic one
for cm in 0 1; do
  for sh in 0 1; do ./build/cuda_emu 8 3 1500 8 $sh $cm; ./build/cuda_emu 64 2 1200 13 $sh $cm; ./build/cuda_emu 100 1 800 32 $sh $cm; done
  ./build/cuda_emu 8 2 600 8 1 $cm 1; ./build/cuda_emu 8 1 4500 8 1 $cm; ./build/cuda_emu 64 1 4300 13 0 $cm
  ./build/cuda_emu_f 8 3 1500 8 1 $cm; ./build/cuda_emu_f 64 2 1200 13 1 $cm; ./build/cuda_emu_f 8 1 4500 8 0 $cm
done
echo "== CUDA kernel compiled to PTX with clang (skipped if clang lacks the NVPTX backend)"
if ${CLANG:-clang} -print-targets 2>/dev/null | grep -q nvptx64; then make -s ptx && echo "PTX sm_70/sm_80 float64/float32: ok"; else echo "skip: no clang with NVPTX"; fi
echo "== digits end-to-end (C++ classifier)"
python3 python/digits_test.py build/hgln_d 3
echo "ALL TESTS PASSED"
