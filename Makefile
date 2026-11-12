CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
BIN = build
all: $(BIN)/hgln_d $(BIN)/hgln_f $(BIN)/test_api $(BIN)/cuda_emu
$(BIN):
	mkdir -p $(BIN)
$(BIN)/hgln_d: src/hgln_cli.cpp include/hgln.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Iinclude $< -o $@
$(BIN)/hgln_f: src/hgln_cli.cpp include/hgln.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -DUSE_FLOAT -Iinclude $< -o $@
$(BIN)/test_api: tests/test_api.cpp include/hgln.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Iinclude $< -o $@
$(BIN)/cuda_emu: cuda/cuda_emu.cpp cuda/hgln_kernel.h cuda/hgln_host.h include/hgln.hpp | $(BIN)
	$(CXX) -O2 -std=c++20 -pthread -Icuda -Iinclude $< -o $@
test: all
	sh scripts/run_tests.sh
clean:
	rm -rf $(BIN)
.PHONY: all test clean
