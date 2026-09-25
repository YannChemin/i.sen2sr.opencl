#ifndef I_SEN2SR_OPENCL_SEN2SR_MODEL_H
#define I_SEN2SR_OPENCL_SEN2SR_MODEL_H

#include "ocl_backend.h"

/* Side of the low-resolution tile the SEN2SR models are trained on. The
 * hard-constraint masks are stored for exactly this size. */
#define SR_TILE 128

/* Sentinel-2 bands in the order used throughout SEN2SR. */
enum s2_band { B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, S2_NBANDS };

enum sr_variant {
    SR_RGBN_X4,  /* B02 B03 B04 B08, 10 m -> 2.5 m. */
    SR_RSWIR_X2, /* 20 m bands -> 10 m, guided by all 10 bands. */
    SR_ALL_X4    /* All 10 bands -> 2.5 m. */
};

struct sr_conv {
    int cin, cout, k;
    cl_mem w, b;
};

/* SEN2SRLite CNN (SPAN), with every Conv3XC folded into one 3x3
 * convolution. Only the first and last attention blocks influence the
 * output (see sen2sr_model.c), so only those are kept. */
struct sr_cnn {
    int cin, cout, feat, up;
    struct sr_conv conv1, first[3], last[3], conv2, cat, ups;
};

/* Fourier low-pass constraint for n x n outputs. */
struct sr_hc {
    int n;
    cl_mem mask; /* ifftshift(mask), (n, n). */
};

struct sr_resize {
    int in, out, kmax;
    cl_mem idx, wts;
};

struct sr_model {
    enum sr_variant variant;
    int scale; /* Output cells per input cell along each axis. */
    int nin, nout;
    enum s2_band in_bands[S2_NBANDS], out_bands[S2_NBANDS];

    struct ocl_backend *ocl;
    cl_kernel k_conv, k_dec, k_rx, k_ry, k_diff, k_mask, k_add, k_fft;

    /* rgbn: RGBN x4 network (the whole model for SR_RGBN_X4).
     * f2: 20 m -> 10 m network (the whole model for SR_RSWIR_X2).
     * fus: 2.5 m fusion network of SR_ALL_X4. */
    struct sr_cnn rgbn, f2, fus;
    struct sr_hc hc_rgbn, hc_f2, hc_fus;
    struct sr_resize cubic4, linear4, linear2, cubic1;

    /* Device buffers for one tile. */
    cl_mem in, x10, x4, sub, f2out, rgbn_sr, fin, fus_out;
    cl_mem ws_cat, ws_a, ws_b, rs_tmp, lr_up, spec;
};

/* Load a SEN2SRLite model directory (as downloaded by mlstac) and
 * detect which variant it holds. Fatal on missing or malformed files and
 * on the non-Lite (Mamba) SEN2SR models, which are not supported. */
void sr_model_load(struct sr_model *m, struct ocl_backend *ocl,
                   const char *dir);

void sr_model_free(struct sr_model *m);

/* Name of the variant, for messages. */
const char *sr_variant_name(enum sr_variant v);

/* Super-resolve one tile. in holds nin planes of SR_TILE x SR_TILE
 * reflectances, ordered as in_bands; out receives nout planes of
 * (SR_TILE * scale)^2, ordered as out_bands. */
void sr_model_run(struct sr_model *m, const float *in, float *out);

#endif /* I_SEN2SR_OPENCL_SEN2SR_MODEL_H */
