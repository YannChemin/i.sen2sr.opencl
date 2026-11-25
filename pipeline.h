#ifndef SEN2SR_OPENCL_PIPELINE_H
#define SEN2SR_OPENCL_PIPELINE_H

#include "mamba_sr.h"
#include "resize.h"
#include "swin2sr.h"

/* Full ten-band SEN2SR x4 reconstruction: RGBN MambaSR + two Swin2SR
 * stages + three Fourier hard constraints. */
struct pipeline {
    struct nn *nn;
    struct mamba_sr rgbn;
    struct swin2sr f2, f4;
    struct resize_map bicubic_same;  /* 128 -> 128, bicubic, antialiased */
    struct resize_map bicubic_x4;    /* 128 -> 512, bicubic, antialiased */
    struct resize_map bilinear_x2;   /* 64  -> 128, bilinear, antialiased */
    struct resize_map bilinear_x4;   /* 128 -> 512, bilinear, antialiased */
    cl_mem hc_rgbn, hc_f2, hc_f4;
    cl_mem spec_rgbn, spec_f2, spec_f4;
};

/* Weights are looked up next to model_dir; use the exact SEN2SR bundle. */
void pipeline_load(struct pipeline *p, struct nn *nn, const char *model_dir);
void pipeline_free(struct pipeline *p);

/* in: 1x10xH xW (S2 order), out: 1x10x(4H)x(4W). */
void pipeline_forward(struct pipeline *p, struct tensor *in,
                      struct tensor *out);

#endif /* SEN2SR_OPENCL_PIPELINE_H */
