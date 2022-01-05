# The definitions in this file should be changed to match your system

SHELL := /bin/bash

CUDA_PATH   ?= /usr/local/cuda
INSTALL_DIR ?= $(HOME)/linux_64

CUDA_DIR   ?= $(CUDA_PATH)
THRUST_DIR ?= $(CUDA_DIR)/include

GPU_ARCH   ?= sm_30

#SYCLCC     ?= /opt/intel/oneapi/compiler/latest/linux/bin/dpcpp
#SYCL_FLAGS ?=

SYCLCC     ?= /opt/hipSYCL/bin/syclcc
SYCL_FLAGS ?= --hipsycl-targets="omp"

#SYCLCC     ?= /opt/computecpp/bin/compute++
#SYCL_FLAGS ?= -sycl-driver -I /opt/computecpp/include/ -L /opt/computecpp/lib/ -lComputeCpp -sycl-target spir64

#SYCLCC     ?= clang-12
#SYCL_FLAGS ?= -I/opt/triSYCL/include -I/opt/mdspan/include -fopenmp -lomp -std=c++17

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
