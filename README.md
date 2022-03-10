Migrating to SYCL, work in progress...

This implementation contains part of modified Intel's DPC++ compatibility tool (DPCT), see `src/dpct`

This implementation uses a forked version of SyclParallelSTL for compatibility with multiple SYCL implementations.

Compiler setups can be found in Makefile.inc .

---

# dedisp
This repositry is derived from Ben Barsdell's original GPU De-dedispersion library (code.google.com/p/dedisp)

Installation Instructions:

  1.  git clone https://github.com/ajameson/dedisp.git
  2.  Update Makefile.inc with your CUDA path, Install Dir and GPU architecture. e.g.
      * CUDA_PATH ?= /usr/local/cuda-8.0.61
      * INSTALL_DIR = $(HOME)/opt/dedisp
      * GPU_ARCH = sm_60
  3.  make && make install
  
  This will build a shared object library named libdedisp.so which is a prerequisite for Heimdall. The dedisp header files will be installed into INSTALL_DIR/include and the library into INSTALL_DIR/lib.
