#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Runs every test that can run on this machine. Needs: g++ (C++20), python3 with numpy and scikit-learn.
set -e
cd "$(dirname "$0")/.."
echo "== C++ behavioural tests"; ./build/test_api
echo "== NumPy vs C++ parity (float64, d=8 and d=64; float32, d=8 and d=64)"
python3 python/parity.py build/hgln_d 8; python3 python/parity.py build/hgln_d 64
python3 python/parity.py build/hgln_f 8; python3 python/parity.py build/hgln_f 64
echo "== CUDA kernel logic on the CPU emulator vs the C++ library (this does NOT run on a GPU)"
for sh in 0 1; do ./build/cuda_emu 8 3 1500 8 $sh; ./build/cuda_emu 64 2 1200 13 $sh; ./build/cuda_emu 100 1 800 32 $sh; done
./build/cuda_emu_f 8 3 1500 8 1; ./build/cuda_emu_f 64 2 1200 13 1
echo "== CUDA kernel compiled to PTX with clang (skipped if clang lacks the NVPTX backend)"
if ${CLANG:-clang} -print-targets 2>/dev/null | grep -q nvptx64; then make -s ptx && echo "PTX sm_70/sm_80 float64/float32: ok"; else echo "skip: no clang with NVPTX"; fi
echo "== digits end-to-end (C++ classifier)"
python3 python/digits_test.py build/hgln_d 3
echo "ALL TESTS PASSED"
