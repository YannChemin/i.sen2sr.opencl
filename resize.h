#ifndef SEN2SR_OPENCL_RESIZE_H
#define SEN2SR_OPENCL_RESIZE_H

#include "ocl_backend.h"

/* Separable anti-aliased resize map as in PyTorch interpolate(...,
 * antialias=True).  idx[2*i] is the first source pixel and idx[2*i+1]
 * the number of taps; wts[i*kmax + j] is the matching normalised weight. */
struct resize_map {
    cl_mem idx_x, wts_x;
    cl_mem idx_y, wts_y;
    int in, out, kmax;
};

void resize_map_build(struct ocl_backend *ocl, struct resize_map *r, int in,
                      int out, int cubic);
void resize_map_free(struct resize_map *r);

/* Fourier low-pass mask for an n x n output, ifftshifted for the
 * natural-order FFT used in the kernels.  When n differs from the stored
 * 512 x 512 mask an ideal filter with radius n/8 is generated instead. */
cl_mem hc_mask_load(struct ocl_backend *ocl, const char *path, int n);

#endif /* SEN2SR_OPENCL_RESIZE_H */
