# The definitions in this file should be changed to match your system

SHELL := /bin/bash

CUDA_PATH   ?= /usr/local/cuda
INSTALL_DIR ?= $(HOME)/linux_64

CUDA_DIR   ?= $(CUDA_PATH)
THRUST_DIR ?= $(CUDA_DIR)/include

GPU_ARCH   ?= sm_30

# Some implementations

# OneAPI distributed clang, missing <sycl/sycl.hpp> header (patched using `__has_include(<sycl/sycl.hpp>)`)
#SYCLCC     ?= /opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang -fsycl
#SYCL_FLAGS ?=

# OneAPI distributed dpcpp, missing <sycl/sycl.hpp> header (patched using `__has_include(<sycl/sycl.hpp>)`). They use <CL/sycl.hpp> and cl::sycl
#SYCLCC     ?= /opt/intel/oneapi/compiler/latest/linux/bin/dpcpp
#SYCL_FLAGS ?=

# hipSYCL syclcc
#SYCLCC     ?= /opt/hipSYCL/bin/syclcc
#SYCL_FLAGS ?= --hipsycl-targets="omp;cuda:sm_60;hip:gfx906" --hipsycl-explicit-multipass --hipsycl-cuda-path=/usr/local/cuda

# hipSYCL syclcc + intel/llvm fork, missing '-fsycl-unnamed-lambda' flag, "highly experimental".
#SYCLCC     ?= /opt/hipSYCL/bin/syclcc
#SYCL_FLAGS ?= --hipsycl-targets="omp;spirv" --hipsycl-clang=/opt/intel-llvm/bin/clang

# Codeplay ComputeCpp, missing SYCL 2020 features
#SYCLCC     ?= /opt/computecpp/bin/compute++
#SYCL_FLAGS ?= -sycl-driver -I /opt/computecpp/include/ -L /opt/computecpp/lib/ -lComputeCpp -sycl-target spir64

# triSYCL/triSYCL, missing SYCL 2020 features
#SYCLCC     ?= clang
#SYCL_FLAGS ?= -I/opt/triSYCL/include -I/opt/mdspan/include -fopenmp -lomp -std=c++17

# intel/llvm
#SYCLCC     ?= /opt/intel-llvm/bin/clang -fsycl
#SYCL_FLAGS ?= -lstdc++

LIB_ARCH   = lib64

GCC        = gcc
GXX        = g++
AR         = ar
# NVCC     = $(CUDA_DIR)/bin/nvcc #-Xptxas -abi=no
DOXYGEN    = doxygen
RM         = rm
ECHO       = echo
MKFLAGS    = 

DEDISP_DEBUG = 1

#.SILENT :
