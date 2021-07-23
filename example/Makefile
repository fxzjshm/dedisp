DEDISP_HOME ?= ..
include $(DEDISP_HOME)/Makefile.inc

testdedisp: testdedisp.c
	source $(ONEAPI_PATH)/setvars.sh && $(GCC) -Wall -O3 -I$(DEDISP_HOME)/src -L$(DEDISP_HOME)/lib -o testdedisp testdedisp.c gasdev.c ran1.c -ldedisp -lm

clean:
	rm -f testdedisp
