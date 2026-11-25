#ifndef I_SEN2SR_OPENCL_SEN2SR_FULL_H
#define I_SEN2SR_OPENCL_SEN2SR_FULL_H

#include "ocl_backend.h"
#include "pipeline.h"
#include "sen2sr_model.h"

/* Full SEN2SR (MambaSR + two Swin2SR stages) wrapped in the same tile
 * interface as the bundled SEN2SRLite models. It is fp32-only and works
 * on 128 x 128 input tiles, producing ten bands at 512 x 512. */
struct sr_full {
    int scale, nin, nout;
    enum s2_band in_bands[S2_NBANDS], out_bands[S2_NBANDS];
    struct nn nn;
    struct pipeline pipe;
    struct tensor *in, *out;
};

/* True when dir holds a complete full SEN2SR bundle, as opposed to the
 * SEN2SRLite files with the same names. Never fatal on missing files. */
int sr_full_detect(const char *dir);

void sr_full_load(struct sr_full *m, struct ocl_backend *ocl, const char *dir);
void sr_full_run(struct sr_full *m, const float *in, float *out);
void sr_full_free(struct sr_full *m);

#endif /* I_SEN2SR_OPENCL_SEN2SR_FULL_H */
