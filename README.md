Migrated to OpenCL using Boost.Compute  
Results of `example/testdedisp` seems passable compared to original CUDA version, though not tested seriously.
  
Installation: just `make && make install` it. If it can't be built, check `CL_INCLUDE_DIR` and `BOOST_COMPUTE_INLCUDE_DIR` in Makefile.inc

Below is the original Readme.

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
