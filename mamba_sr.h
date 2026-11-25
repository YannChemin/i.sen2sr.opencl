#ifndef SEN2SR_OPENCL_MAMBA_SR_H
#define SEN2SR_OPENCL_MAMBA_SR_H

#include "nn.h"

struct mamba_cfg {
    int H, W, in_ch, out_ch, C, D, groups, depth, up, dt_rank, state;
};

struct mamba_block {
    struct tensor *ln1w, *ln1b;
    struct tensor *in_w;
    struct tensor *conv_w, *conv_b;
    struct tensor *xproj_w, *dt_w, *dt_b, *A, *Ds;
    struct tensor *outw, *outb, *outproj_w;
    struct tensor *ln2w, *ln2b;
    struct tensor *cab0w, *cab0b, *cab2w, *cab2b;
    struct tensor *att1w, *att1b, *att3w, *att3b;
    struct tensor *skip1, *skip2;
};

struct mamba_layer {
    struct mamba_block *blocks;
    struct tensor *conv_w, *conv_b;
};

struct mamba_sr {
    struct nn *nn;
    struct mamba_cfg cfg;
    struct tensor *conv_first_w, *conv_first_b;
    struct tensor *pe_norm_w, *pe_norm_b;
    struct tensor *norm_w, *norm_b;
    struct tensor *conv_body_w, *conv_body_b;
    struct tensor *conv_before_w, *conv_before_b;
    struct tensor *up0_w, *up0_b, *up2_w, *up2_b;
    struct tensor *conv_last_w, *conv_last_b;
    struct mamba_layer *layer;

    int cur_H, cur_W;
    struct tensor *tok[3], *group_in, *xz, *xx, *zz, *feat, *feat2, *first;
    struct tensor *body, *body2;
    struct tensor *xs, *xdbl, *dts6, *Bs, *Cs, *dts, *out_y, *y, *yn, *yz;
    struct tensor *ss, *cab1, *cab2, *cab_out, *pool, *pool2, *att3;
    struct tensor *pre_up, *up1, *up2;
};

void mamba_sr_load(struct mamba_sr *m, struct nn *nn, const char *file,
                   struct mamba_cfg cfg);
void mamba_sr_free(struct mamba_sr *m);
void mamba_sr_forward(struct mamba_sr *m, struct tensor *in,
                      struct tensor *out);

#endif /* SEN2SR_OPENCL_MAMBA_SR_H */
