# The definitions in this file should be changed to match your system

SHELL := /bin/bash

CUDA_PATH   ?= /usr/local/cuda
INSTALL_DIR ?= $(HOME)/linux_64

CUDA_DIR   ?= $(CUDA_PATH)
THRUST_DIR ?= $(CUDA_DIR)/include

GPU_ARCH   ?= sm_30

ONEAPI_PATH ?= /opt/intel/oneapi

LIB_ARCH   = lib64

GCC        = gcc
GXX        = g++
AR         = ar
# NVCC     = $(CUDA_DIR)/bin/nvcc #-Xptxas -abi=no
DPCPP      = $(ONEAPI_PATH)/compiler/latest/linux/bin/dpcpp
DOXYGEN    = doxygen
RM         = rm
ECHO       = echo
MKFLAGS    = 

DEDISP_DEBUG = 0

#.SILENT :
