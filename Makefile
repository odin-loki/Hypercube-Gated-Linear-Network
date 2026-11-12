# SPDX-License-Identifier: AGPL-3.0-or-later
# make            build the CLI (float64/float32), tests and the CUDA emulator into build/
# make test       run every test that can run without a GPU (about 15 seconds)
# make ptx        compile the CUDA kernel to PTX with clang (no CUDA SDK needed), sm_70 and sm_80, float64 and float32
# make bench      the speed benchmarks quoted in docs/OPTIMISATION.md
# ARCH=-march=x86-64-v3 (or any -march) builds a binary for other machines; the default tunes for this CPU.
CXX ?= g++
CLANG ?= clang
ARCH ?= -march=native
CXXFLAGS ?= -O3 $(ARCH) -std=c++17 -Wall -Wextra -pthread
BIN = build
LIB = include/hgln.hpp include/hgln_gate.h
KERN = cuda/hgln_kernel.h cuda/hgln_dims.h cuda/hgln_host.h include/hgln_gate.h
PTXFLAGS = -x cuda --cuda-device-only -nocudainc -nocudalib -Wno-unknown-cuda-version -O3 -ffp-contract=off -Wall -Werror -Icuda -Iinclude -S
all: $(BIN)/hgln_d $(BIN)/hgln_f $(BIN)/test_api $(BIN)/cuda_emu $(BIN)/cuda_emu_f
$(BIN):
	mkdir -p $(BIN)
$(BIN)/hgln_d: src/hgln_cli.cpp $(LIB) | $(BIN)
	$(CXX) $(CXXFLAGS) -Iinclude $< -o $@
$(BIN)/hgln_f: src/hgln_cli.cpp $(LIB) | $(BIN)
	$(CXX) $(CXXFLAGS) -DUSE_FLOAT -Iinclude $< -o $@
$(BIN)/test_api: tests/test_api.cpp $(LIB) | $(BIN)
	$(CXX) $(CXXFLAGS) -Iinclude $< -o $@
$(BIN)/cuda_emu: cuda/cuda_emu.cpp $(KERN) $(LIB) | $(BIN)
	$(CXX) -O2 -std=c++20 -pthread -Wno-psabi -Icuda -Iinclude $< -o $@
$(BIN)/cuda_emu_f: cuda/cuda_emu.cpp $(KERN) $(LIB) | $(BIN)
	$(CXX) -O2 -std=c++20 -pthread -Wno-psabi -DHG_REAL=float -Icuda -Iinclude $< -o $@
ptx: cuda/hgln_ptx.cu cuda/hgln_dims.h $(KERN) | $(BIN)
	$(CLANG) $(PTXFLAGS) --cuda-gpu-arch=sm_70 cuda/hgln_ptx.cu -o $(BIN)/hgln.sm70.ptx
	$(CLANG) $(PTXFLAGS) --cuda-gpu-arch=sm_80 cuda/hgln_ptx.cu -o $(BIN)/hgln.sm80.ptx
	$(CLANG) $(PTXFLAGS) --cuda-gpu-arch=sm_70 -DHG_REAL=float cuda/hgln_ptx.cu -o $(BIN)/hgln_f.sm70.ptx
	$(CLANG) $(PTXFLAGS) --cuda-gpu-arch=sm_80 -DHG_REAL=float cuda/hgln_ptx.cu -o $(BIN)/hgln_f.sm80.ptx
	@! grep -l '\.extern \.func' $(BIN)/*.ptx || (echo "PTX has unresolved external calls" && false)
test: all
	sh scripts/run_tests.sh
bench: $(BIN)/hgln_d $(BIN)/hgln_f
	./$(BIN)/hgln_d bench 8 1 20000 1; ./$(BIN)/hgln_d bench 64 10 3000; ./$(BIN)/hgln_f bench 64 10 3000
clean:
	rm -rf $(BIN)
.PHONY: all test clean ptx bench
