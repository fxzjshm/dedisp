
include Makefile.inc

# Output directories
BIN_DIR     = bin
OBJ_DIR     = obj
LIB_DIR     = lib
INCLUDE_DIR = include

SRC_DIR   := src
#INC_DIR   := ./include
OPTIMISE  := -O3
# Note: Using -G makes the GPU kernel 16x slower!
DEBUG     := -g -DDEDISP_DEBUG=$(DEDISP_DEBUG) #-G

INCLUDE   := -I$(SRC_DIR) -I$(CL_INCLUDE_DIR) -I$(BOOST_COMPUTE_INLCUDE_DIR)
# LIB       := -L$(CUDA_DIR)/$(LIB_ARCH) -lcudart -lstdc++
LIB       := -lstdc++ -lOpenCL

SOURCES   := $(SRC_DIR)/dedisp.dp.cpp
HEADERS   := $(SRC_DIR)/dedisp.h $(SRC_DIR)/kernels.dp.hpp         \
             $(SRC_DIR)/gpu_memory.hpp $(SRC_DIR)/transpose.hpp
INTERFACE := $(SRC_DIR)/dedisp.h
CPP_INTERFACE := $(SRC_DIR)/DedispPlan.hpp
CL_SOURCES := $(SRC_DIR)/transpose.cl $(SRC_DIR)/dedisperse.cl $(SRC_DIR)/unpack.cl $(SRC_DIR)/scrunch_x2.cl
CL_XXD_FILES := $(patsubst $(SRC_DIR)/%.cl, $(SRC_DIR)/%.cl.xxd.txt, $(CL_SOURCES))

LIB_NAME  := libdedisp
SO_EXT    := .so
A_EXT     := .a
MAJOR     := 1
MINOR     := 0.1
SO_FILE   := $(LIB_NAME)$(SO_EXT).$(MAJOR).$(MINOR)
SO_NAME   := $(LIB_DIR)/$(SO_FILE)
A_NAME    := $(LIB_DIR)/$(LIB_NAME)$(A_EXT)
LD_FLAGS  := -shared -fPIC -Wl,--version-script=libdedisp.version,-soname,$(LIB_NAME)$(SO_EXT).$(MAJOR)

# PTX_NAME  := ./dedisp_kernels.ptx

all: shared

#$(ECHO) Building shared library $(SO_FILE)
shared: $(SO_NAME)

$(SO_NAME): $(SOURCES) $(HEADERS) $(CL_XXD_FILES)
	mkdir -p $(INCLUDE_DIR)
	mkdir -p $(LIB_DIR)
	mkdir -p $(OBJ_DIR)
	$(CLCC) -std=$(CXX_STD) $(LD_FLAGS) $(OPTIMISE) $(DEBUG) $(INCLUDE) -o $(SO_NAME) $(SOURCES) $(LIB)
	ln -s -f $(SO_FILE) $(LIB_DIR)/$(LIB_NAME)$(SO_EXT).$(MAJOR)
	ln -s -f $(SO_FILE) $(LIB_DIR)/$(LIB_NAME)$(SO_EXT)
	cp $(INTERFACE) $(INCLUDE_DIR)
	cp $(CPP_INTERFACE) $(INCLUDE_DIR)

$(SRC_DIR)/%.cl.xxd.txt: $(SRC_DIR)/%.cl
	xxd -i < $< > $@ && echo ', 0' >> $@

#static: $(A_NAME)

#$(A_NAME): $(SRC_DIR)/dedisp.cu $(HEADERS)
#	$(NVCC) -c -Xcompiler "-fPIC -Wall" -arch=$(GPU_ARCH) $(OPTIMISE) $(DEBUG) -o $(OBJ_DIR)/dedisp.o $(SRC_DIR)/dedisp.cu
#	$(AR) rcs $(A_NAME) $(OBJ_DIR)/dedisp.o
#	cp $(INTERFACE) $(INCLUDE_DIR)
#	cp $(CPP_INTERFACE) $(INCLUDE_DIR)

test: $(SO_NAME)
	cd test; $(MAKE) $(MKARGS)

# ptx: $(PTX_NAME)

# $(PTX_NAME): $(SOURCES) $(LIB_DIR)/libdedisp.so $(HEADERS)
# 	$(NVCC) -ptx -Xcompiler "-fPIC -Wall" $(OPTIMISE) $(DEBUG) -arch=$(GPU_ARCH) $(INCLUDE) -o $(PTX_NAME) $(SRC_DIR)/dedisp.cu

doc: $(SRC_DIR)/dedisp.h Doxyfile
	$(DOXYGEN) Doxyfile

clean:
	$(RM) -f $(SO_NAME) $(A_NAME) $(OBJ_DIR)/*.o $(LIB_DIR)/*.so $(LIB_DIR)/*.so.1 $(SRC_DIR)/*.cl.xxd.txt

install: all
	cp $(INTERFACE) $(INSTALL_DIR)/include/
	cp $(CPP_INTERFACE) $(INSTALL_DIR)/include/
	cp $(SO_NAME) $(INSTALL_DIR)/lib/
	ln -s -f $(SO_FILE) $(INSTALL_DIR)/lib/$(LIB_NAME)$(SO_EXT).$(MAJOR)
	ln -s -f $(SO_FILE) $(INSTALL_DIR)/lib/$(LIB_NAME)$(SO_EXT)
