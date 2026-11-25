#ifndef SEN2SR_OPENCL_SWINSR_H
#define SEN2SR_OPENCL_SWINSR_H

#include "nn.h"

struct swin_cfg {
    int H, W, C, heads, window, depth, layers, mlp_ratio, out_ch;
};

struct swin_maps {
    cl_mem in, out; /* int32 maps of L*C elements */
};

struct swin_block {
    struct tensor *n1w, *n1b;
    struct tensor *qkv_w, *qkv_b;
    struct tensor *rel;  /* (heads, N, N), computed from cpb weights */
    struct tensor *logit;/* (heads) */
    struct tensor *proj_w, *proj_b;
    struct tensor *mask; /* (nW, N, N), rebuilt per input size */
    struct tensor *n2w, *n2b;
    struct tensor *fc1w, *fc1b, *fc2w, *fc2b;
    int shifted;
};

struct swin_layer {
    struct tensor *conv_w, *conv_b;
    struct swin_block *blocks;
};

struct swin2sr {
    struct nn *nn;
    struct swin_cfg cfg;
    int N, nW, d;

    struct tensor *conv_first_w, *conv_first_b;
    struct tensor *pe_proj_w, *pe_proj_b;
    struct tensor *pe_norm_w, *pe_norm_b;
    struct tensor *norm_w, *norm_b;
    struct tensor *conv_body_w, *conv_body_b;
    struct tensor *conv_before_w, *conv_before_b;
    struct tensor *conv_last_w, *conv_last_b;

    struct swin_layer *layer;
    struct swin_maps maps[2];

    int cur_H, cur_W;
    struct tensor *tok[3], *win, *qkv, *q, *k, *v, *attn, *ctx, *proj;
    struct tensor *ffn1, *ffn2, *feat, *feat2, *feat64;
};

void swin2sr_load(struct swin2sr *m, struct nn *nn, const char *file,
                  struct swin_cfg cfg);
void swin2sr_free(struct swin2sr *m);
void swin2sr_forward(struct swin2sr *m, struct tensor *in, struct tensor *out);

#endif /* SEN2SR_OPENCL_SWINSR_H */
