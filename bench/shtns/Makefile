# Native SHTNS benchmarks, to compare against the WGSL transforms in ../../src/sht.
#
#   ./bootstrap.sh     # fetch and build upstream SHTNS, write shtns.mk
#   make               # shtbench, plus shtbench_gpu if SHTNS has CUDA support
#
# -ffast-math is deliberately NOT used: the fp32 GPU run is compared element by
# element against the WebGPU one, and reassociation would make that a comparison
# of two different computations. -O3 without it is what SHTNS' own codelets use.

-include shtns.mk

ifeq ($(SHTNS_LIB),)
$(error run ./bootstrap.sh first — it fetches SHTNS and writes shtns.mk)
endif

CXX ?= g++
NVCC ?= nvcc
# Compile for the GPUs in this machine, which is both the fastest build and the
# right answer for a benchmark that only ever runs here. Needs CUDA 11.5+; on an
# older toolkit, or to build elsewhere and run here, say
# `make NVCC_ARCH=-arch=sm_80`.
NVCC_ARCH ?= -arch=native

CXXFLAGS ?= -O3 -march=native -std=c++14 -fopenmp -Wall -Wextra -Wno-unused-parameter
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo
INCLUDES = -I$(SHTNS_DIR)
LDLIBS = $(SHTNS_LIB) $(SHTNS_LDLIBS)

TARGETS = shtbench
ifeq ($(SHTNS_HAS_GPU),1)
TARGETS += shtbench_gpu
endif

all: $(TARGETS)
	@echo
	@echo "built: $(TARGETS)"
ifneq ($(SHTNS_HAS_GPU),1)
	@echo "note: SHTNS was built without GPU support, so there is no shtbench_gpu."
	@echo "      Re-run ./bootstrap.sh on a machine with nvcc to get it."
endif

shtbench: shtbench.cpp spec.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDLIBS)

# nvcc drives the link so the CUDA runtime and the C++ standard library (which
# SHTNS' .cu objects need) come along without naming them by hand.
shtbench_gpu: shtbench_gpu.cu spec.h
	$(NVCC) $(NVCCFLAGS) $(NVCC_ARCH) $(INCLUDES) $< -o $@ \
	  -Xcompiler -fopenmp $(LDLIBS)

clean:
	rm -f shtbench shtbench_gpu

# Also throws away the SHTNS checkout and shtns.mk; ./bootstrap.sh rebuilds them.
distclean: clean
	rm -rf $(SHTNS_DIR) shtns.mk

.PHONY: all clean distclean
