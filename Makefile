MODULE_TOPDIR = ../..

PGM = i.sen2sr.opencl

LIBES = $(RASTERLIB) $(GISLIB) $(MATHLIB)
DEPENDENCIES = $(RASTERDEP) $(GISDEP)
EXTRA_INC = $(OCLINCPATH) -I$(OBJDIR)
EXTRA_CFLAGS = -O3 -std=gnu11
EXTRA_LIBS = $(OCLLIBPATH) $(OCLLIB)

# The three SEN2SRLite model variants ship with the module and are
# installed under $(ETC)/$(PGM)/models/<variant>/.
ETCFILES = $(wildcard models/*/*.safetensor) models/README.md

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

# Module.make installs ETCFILES flat into $(ETC)/$(PGM); the model files
# live in per-variant subdirectories, which must be created first. The
# stem of this rule is shorter, so make prefers it for these files.
$(ETC)/$(PGM)/models/%: models/%
	$(MKDIR) $(dir $@)
	$(INSTALL_DATA) $< $@

# Embed the OpenCL kernels as a C string literal.
$(OBJDIR)/ocl_backend.o: $(OBJDIR)/sen2sr_kernels_cl.h

$(OBJDIR)/sen2sr_kernels_cl.h: sen2sr_kernels.cl | $(OBJDIR)
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' $< > $@
