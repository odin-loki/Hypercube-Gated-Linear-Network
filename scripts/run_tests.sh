#!/bin/sh
# Runs every test that can run on this machine. Needs: g++ (C++20), python3 with numpy and scikit-learn.
set -e
cd "$(dirname "$0")/.."
echo "== C++ behavioural tests"; ./build/test_api
echo "== NumPy vs C++ parity (float64, d=8 and d=64; float32, d=8 and d=64)"
python3 python/parity.py build/hgln_d 8; python3 python/parity.py build/hgln_d 64
python3 python/parity.py build/hgln_f 8; python3 python/parity.py build/hgln_f 64
echo "== CUDA kernel logic on the CPU emulator vs reference C++ (this does NOT run on a GPU)"
./build/cuda_emu 8 3 1500 8; ./build/cuda_emu 64 2 1200 13; ./build/cuda_emu 100 1 800 32
echo "== digits end-to-end (C++ classifier)"
python3 python/digits_test.py build/hgln_d 3
echo "ALL TESTS PASSED"
