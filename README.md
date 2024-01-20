# dedisp
This repositry is derived from Andrew Jameson's dedisp, which is derived from Ben Barsdell's original GPU De-dedispersion library (code.google.com/p/dedisp)

Installation Instructions:

  1.  git clone https://github.com/ajameson/dedisp.git
  2.  Update Makefile.inc with your ROCm path, Install Dir and GPU architecture. e.g.
      * CUDA_PATH ?= /opt/rocm
      * INSTALL_DIR = $(HOME)/opt/dedisp
      * GPU_ARCH = gfx906
  3.  make && make install
  
  This will build a shared object library named libdedisp.so which is a prerequisite for Heimdall. The dedisp header files will be installed into INSTALL_DIR/include and the library into INSTALL_DIR/lib.
