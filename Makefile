MODULE_TOPDIR = ../..

PGM = i.sen2sr.opencl

LIBES = $(RASTERLIB) $(GISLIB) $(MATHLIB)
DEPENDENCIES = $(RASTERDEP) $(GISDEP)
EXTRA_INC = $(OCLINCPATH) -I$(OBJDIR)
EXTRA_CFLAGS = -O3 -std=gnu11
EXTRA_LIBS = $(OCLLIBPATH) $(OCLLIB)

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

# Embed the OpenCL kernels as a C string literal.
$(OBJDIR)/ocl_backend.o: $(OBJDIR)/sen2sr_kernels_cl.h

$(OBJDIR)/sen2sr_kernels_cl.h: sen2sr_kernels.cl | $(OBJDIR)
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' $< > $@
