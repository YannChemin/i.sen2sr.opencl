#include "mamba_sr.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct tensor *ld1(struct ocl_backend *ocl, struct st_file *f,
                          const char *name, int n)
{
    long s[1] = {n};
    return tensor_load(ocl, f, name, 1, s);
}

static struct tensor *ld2(struct ocl_backend *ocl, struct st_file *f,
                          const char *name, int a, int b)
{
    long s[2] = {a, b};
    return tensor_load(ocl, f, name, 2, s);
}

static struct tensor *ld4(struct ocl_backend *ocl, struct st_file *f,
                          const char *name, int a, int b, int k)
{
    long s[4] = {a, b, k, k};
    return tensor_load(ocl, f, name, 4, s);
}

/* A = -exp(A_logs) as expected by the selective scan kernel. */
static struct tensor *load_A(struct ocl_backend *ocl, struct st_file *f,
                             const char *name, int rows, int state)
{
    long s[2] = {rows, state};
    struct tensor *src = tensor_load(ocl, f, name, 2, s);
    float *h = malloc((size_t)rows * state * sizeof(float));
    struct tensor *dst;
    int i;

    if (!h)
        die("out of memory");
    tensor_read(ocl, src, h);
    for (i = 0; i < rows * state; i++)
        h[i] = -expf(h[i]);
    dst = tensor_upload(ocl, h, 1, rows, state, 1);
    free(h);
    tensor_free(src);
    return dst;
}

static void free_block(struct mamba_block *b)
{
    tensor_free(b->ln1w); tensor_free(b->ln1b); tensor_free(b->in_w);
    tensor_free(b->conv_w); tensor_free(b->conv_b);
    tensor_free(b->xproj_w); tensor_free(b->dt_w); tensor_free(b->dt_b);
    tensor_free(b->A); tensor_free(b->Ds);
    tensor_free(b->outw); tensor_free(b->outb); tensor_free(b->outproj_w);
    tensor_free(b->ln2w); tensor_free(b->ln2b);
    tensor_free(b->cab0w); tensor_free(b->cab0b);
    tensor_free(b->cab2w); tensor_free(b->cab2b);
    tensor_free(b->att1w); tensor_free(b->att1b);
    tensor_free(b->att3w); tensor_free(b->att3b);
    tensor_free(b->skip1); tensor_free(b->skip2);
}

static void free_scratch(struct mamba_sr *m)
{
    int i;
    for (i = 0; i < 3; i++) tensor_free(m->tok[i]);
    tensor_free(m->group_in);
    tensor_free(m->xz); tensor_free(m->xx); tensor_free(m->zz);
    tensor_free(m->feat); tensor_free(m->feat2); tensor_free(m->first);
    tensor_free(m->body); tensor_free(m->body2);
    tensor_free(m->xs); tensor_free(m->xdbl); tensor_free(m->dts6);
    tensor_free(m->Bs); tensor_free(m->Cs); tensor_free(m->dts);
    tensor_free(m->out_y); tensor_free(m->y); tensor_free(m->yn);
    tensor_free(m->yz); tensor_free(m->ss); tensor_free(m->cab1);
    tensor_free(m->cab2); tensor_free(m->cab_out);
    tensor_free(m->pool); tensor_free(m->pool2); tensor_free(m->att3);
    tensor_free(m->pre_up);
    tensor_free(m->up1); tensor_free(m->up2);
}

static void prepare(struct mamba_sr *m, int H, int W)
{
    struct ocl_backend *ocl = m->nn->ocl;
    int L = H * W, C = m->cfg.C, D = m->cfg.D;
    int K = 4, R = m->cfg.dt_rank, S = m->cfg.state;
    int i;

    if (m->cur_H == H && m->cur_W == W)
        return;
    free_scratch(m);
    m->cur_H = H;
    m->cur_W = W;
    for (i = 0; i < 3; i++)
        m->tok[i] = tensor_alloc(ocl, 1, L, C, 1);
    m->group_in = tensor_alloc(ocl, 1, L, C, 1);
    m->xz = tensor_alloc(ocl, 1, L, 2 * D, 1);
    m->xx = tensor_alloc(ocl, 1, L, D, 1);
    m->zz = tensor_alloc(ocl, 1, L, D, 1);
    m->feat = tensor_alloc(ocl, 1, D, H, W);
    m->feat2 = tensor_alloc(ocl, 1, D, H, W);
    m->first = tensor_alloc(ocl, 1, C, H, W);
    m->body = tensor_alloc(ocl, 1, C, H, W);
    m->body2 = tensor_alloc(ocl, 1, C, H, W);
    m->xs = tensor_alloc(ocl, 1, K * D, L, 1);
    m->xdbl = tensor_alloc(ocl, 1, K, R + 2 * S, L);
    m->dts6 = tensor_alloc(ocl, 1, K * R, L, 1);
    m->Bs = tensor_alloc(ocl, 1, K * S, L, 1);
    m->Cs = tensor_alloc(ocl, 1, K * S, L, 1);
    m->dts = tensor_alloc(ocl, 1, K * D, L, 1);
    m->out_y = tensor_alloc(ocl, 1, K * D, L, 1);
    m->y = tensor_alloc(ocl, 1, L, D, 1);
    m->yn = tensor_alloc(ocl, 1, L, D, 1);
    m->yz = tensor_alloc(ocl, 1, L, D, 1);
    m->ss = tensor_alloc(ocl, 1, L, C, 1);
    m->cab1 = tensor_alloc(ocl, 1, 32, H, W);
    m->cab2 = tensor_alloc(ocl, 1, C, H, W);
    m->cab_out = tensor_alloc(ocl, 1, C, H, W);
    m->pool = tensor_alloc(ocl, 1, 1, C, 1);
    m->pool2 = tensor_alloc(ocl, 1, 1, C, 1);
    m->att3 = tensor_alloc(ocl, 1, 1, 3, 1);
    m->pre_up = tensor_alloc(ocl, 1, 64, H, W);
    m->up1 = tensor_alloc(ocl, 1, 64, H * 2, W * 2);
    m->up2 = tensor_alloc(ocl, 1, 64, H * 4, W * 4);
}

void mamba_sr_load(struct mamba_sr *m, struct nn *nn, const char *file,
                   struct mamba_cfg cfg)
{
    struct ocl_backend *ocl = nn->ocl;
    struct st_file f;
    char fmt[256];
    int l, b;

    memset(m, 0, sizeof(*m));
    m->nn = nn;
    m->cfg = cfg;
    st_open(&f, file);

#define L1(field, name, n) ((field) = ld1(ocl, &f, name, n))
#define L2(field, name, a, b_) ((field) = ld2(ocl, &f, name, a, b_))
#define L4(field, name, a, b_, k) ((field) = ld4(ocl, &f, name, a, b_, k))
    L4(m->conv_first_w, "conv_first.weight", cfg.C, cfg.in_ch, 3);
    L1(m->conv_first_b, "conv_first.bias", cfg.C);
    L1(m->pe_norm_w, "patch_embed.norm.weight", cfg.C);
    L1(m->pe_norm_b, "patch_embed.norm.bias", cfg.C);
    L1(m->norm_w, "norm.weight", cfg.C);
    L1(m->norm_b, "norm.bias", cfg.C);
    L4(m->conv_body_w, "conv_after_body.weight", cfg.C, cfg.C, 3);
    L1(m->conv_body_b, "conv_after_body.bias", cfg.C);
    L4(m->conv_before_w, "conv_before_upsample.0.weight", 64, cfg.C, 3);
    L1(m->conv_before_b, "conv_before_upsample.0.bias", 64);
    L4(m->up0_w, "upsample.0.weight", 256, 64, 3);
    L1(m->up0_b, "upsample.0.bias", 256);
    L4(m->up2_w, "upsample.2.weight", 256, 64, 3);
    L1(m->up2_b, "upsample.2.bias", 256);
    L4(m->conv_last_w, "conv_last.weight", cfg.out_ch, 64, 3);
    L1(m->conv_last_b, "conv_last.bias", cfg.out_ch);

    m->layer = calloc(cfg.groups, sizeof(*m->layer));
    if (!m->layer)
        die("out of memory");
    for (l = 0; l < cfg.groups; l++) {
        struct mamba_layer *lay = &m->layer[l];
        lay->blocks = calloc(cfg.depth, sizeof(*lay->blocks));
        if (!lay->blocks)
            die("out of memory");
        snprintf(fmt, sizeof(fmt), "layers.%d.conv.weight", l);
        L4(lay->conv_w, fmt, cfg.C, cfg.C, 3);
        snprintf(fmt, sizeof(fmt), "layers.%d.conv.bias", l);
        L1(lay->conv_b, fmt, cfg.C);
        for (b = 0; b < cfg.depth; b++) {
            struct mamba_block *blk = &lay->blocks[b];
            char n[256];

#define PFX "layers.%d.residual_group.blocks.%d."
#define B1(field, suffix, n_) do { \
            snprintf(n, sizeof(n), PFX suffix, l, b); \
            L1(blk->field, n, n_); \
        } while (0)
#define B2(field, suffix, a_, b_) do { \
            snprintf(n, sizeof(n), PFX suffix, l, b); \
            L2(blk->field, n, a_, b_); \
        } while (0)
#define B4(field, suffix, a_, b_, k_) do { \
            snprintf(n, sizeof(n), PFX suffix, l, b); \
            L4(blk->field, n, a_, b_, k_); \
        } while (0)

            B1(ln1w, "ln_1.weight", cfg.C);
            B1(ln1b, "ln_1.bias", cfg.C);
            B2(in_w, "self_attention.in_proj.weight", 2 * cfg.D, cfg.C);
            B4(conv_w, "self_attention.conv2d.weight", cfg.D, 1, 3);
            B1(conv_b, "self_attention.conv2d.bias", cfg.D);
            snprintf(n, sizeof(n), PFX "self_attention.x_proj_weight", l, b);
            {
                long s[3] = {4, cfg.dt_rank + 2 * cfg.state, cfg.D};
                blk->xproj_w = tensor_load(ocl, &f, n, 3, s);
            }
            snprintf(n, sizeof(n), PFX "self_attention.dt_projs_weight", l, b);
            {
                long s[3] = {4, cfg.D, cfg.dt_rank};
                blk->dt_w = tensor_load(ocl, &f, n, 3, s);
            }
            B2(dt_b, "self_attention.dt_projs_bias", 4, cfg.D);
            snprintf(n, sizeof(n), PFX "self_attention.A_logs", l, b);
            blk->A = load_A(ocl, &f, n, 4 * cfg.D, cfg.state);
            B1(Ds, "self_attention.Ds", 4 * cfg.D);
            B1(outw, "self_attention.out_norm.weight", cfg.D);
            B1(outb, "self_attention.out_norm.bias", cfg.D);
            B2(outproj_w, "self_attention.out_proj.weight", cfg.C, cfg.D);
            B1(ln2w, "ln_2.weight", cfg.C);
            B1(ln2b, "ln_2.bias", cfg.C);
            B4(cab0w, "conv_blk.cab.0.weight", 32, cfg.C, 3);
            B1(cab0b, "conv_blk.cab.0.bias", 32);
            B4(cab2w, "conv_blk.cab.2.weight", cfg.C, 32, 3);
            B1(cab2b, "conv_blk.cab.2.bias", cfg.C);
            B4(att1w, "conv_blk.cab.3.attention.1.weight", 3, cfg.C, 1);
            B1(att1b, "conv_blk.cab.3.attention.1.bias", 3);
            B4(att3w, "conv_blk.cab.3.attention.3.weight", cfg.C, 3, 1);
            B1(att3b, "conv_blk.cab.3.attention.3.bias", cfg.C);
            B1(skip1, "skip_scale", cfg.C);
            B1(skip2, "skip_scale2", cfg.C);
#undef B1
#undef B2
#undef B4
        }
    }
#undef L1
#undef L2
#undef L4
    st_close(&f);
}

void mamba_sr_free(struct mamba_sr *m)
{
    int l, b;
    free_scratch(m);
    tensor_free(m->conv_first_w); tensor_free(m->conv_first_b);
    tensor_free(m->pe_norm_w); tensor_free(m->pe_norm_b);
    tensor_free(m->norm_w); tensor_free(m->norm_b);
    tensor_free(m->conv_body_w); tensor_free(m->conv_body_b);
    tensor_free(m->conv_before_w); tensor_free(m->conv_before_b);
    tensor_free(m->up0_w); tensor_free(m->up0_b);
    tensor_free(m->up2_w); tensor_free(m->up2_b);
    tensor_free(m->conv_last_w); tensor_free(m->conv_last_b);
    if (m->layer) {
        for (l = 0; l < m->cfg.groups; l++) {
            tensor_free(m->layer[l].conv_w);
            tensor_free(m->layer[l].conv_b);
            for (b = 0; b < m->cfg.depth; b++)
                free_block(&m->layer[l].blocks[b]);
            free(m->layer[l].blocks);
        }
        free(m->layer);
    }
}

/* SS2D forward, with input token-major (L,C) and output token-major. */
static void ss2d(struct mamba_sr *m, struct mamba_block *b,
                 struct tensor *in, struct tensor *out)
{
    struct nn *nn = m->nn;
    int L = in->c, C = m->cfg.C, D = m->cfg.D;
    int H = m->cur_H, W = m->cur_W, S = m->cfg.state;
    int R = m->cfg.dt_rank;

    nn_gemm(nn, in, b->in_w, NULL, m->xz, L, C, 2 * D);
    nn_split2(nn, m->xz, m->xx, m->zz, L, D);
    nn_tokens2nchw(nn, m->xx, m->feat, D, L);
    nn_conv(nn, m->feat, m->feat2, b->conv_w, b->conv_b, D, ACT_SILU, 1, NULL);
    nn_make_xs(nn, m->feat2, m->xs, D, H, W);
    nn_proj_xdbl(nn, m->xs, b->xproj_w, m->xdbl, 4, D, R + 2 * S, L);
    nn_split_xdbl(nn, m->xdbl, m->dts6, m->Bs, m->Cs, 4, R, S, L);
    nn_proj_dts(nn, m->dts6, b->dt_w, m->dts, 4, D, R, L);
    nn_scan(nn, m->xs, m->dts, b->A, m->Bs, m->Cs, b->Ds, b->dt_b,
            m->out_y, D, S, L);
    nn_combine_y(nn, m->out_y, m->y, D, H, W);
    nn_layernorm(nn, m->y, m->yn, b->outw, b->outb, L, D, 1e-5f);
    nn_act(nn, m->zz, m->zz, ACT_SILU, 0.0f);
    nn_mul_last(nn, m->yn, m->zz, m->yz, D, L);
    nn_gemm(nn, m->yz, b->outproj_w, NULL, out, L, D, C);
}

static void cab(struct mamba_sr *m, struct mamba_block *b,
                struct tensor *in_nchw, struct tensor *out_nchw)
{
    struct nn *nn = m->nn;
    int C = m->cfg.C, H = m->cur_H, W = m->cur_W, HW = H * W;

    nn_conv(nn, in_nchw, m->cab1, b->cab0w, b->cab0b, 1, ACT_GELU, 1, NULL);
    nn_conv(nn, m->cab1, m->cab2, b->cab2w, b->cab2b, 1, ACT_NONE, 1, NULL);
    nn_pool1(nn, m->cab2, m->pool, C, HW);
    nn_gemm(nn, m->pool, b->att1w, b->att1b, m->att3, 1, C, 3);
    nn_act(nn, m->att3, m->att3, ACT_SIGMOID, 0.0f);
    nn_gemm(nn, m->att3, b->att3w, b->att3b, m->pool2, 1, 3, C);
    nn_act(nn, m->pool2, m->pool2, ACT_SIGMOID, 0.0f);
    nn_mul_bcast(nn, m->cab2, m->pool2, out_nchw, C, HW);
}

void mamba_sr_forward(struct mamba_sr *m, struct tensor *in,
                      struct tensor *out)
{
    struct nn *nn = m->nn;
    int C = m->cfg.C, L = in->h * in->w, H = in->h, W = in->w;
    int l, b;
    struct tensor *x, *xn, *next, *grp;

    prepare(m, H, W);
    nn_conv(nn, in, m->first, m->conv_first_w, m->conv_first_b,
            1, ACT_NONE, 1, NULL);
    nn_nchw2tokens(nn, m->first, m->tok[0], C, L);
    nn_layernorm(nn, m->tok[0], m->tok[1], m->pe_norm_w, m->pe_norm_b,
                 L, C, 1e-5f);
    x = m->tok[1];
    xn = m->tok[0];
    next = m->tok[2];

    for (l = 0; l < m->cfg.groups; l++) {
        nn_lincomb(nn, x, NULL, m->group_in, 1.0f, 0.0f, 0.0f);
        grp = m->group_in;
        for (b = 0; b < m->cfg.depth; b++) {
            struct mamba_block *blk = &m->layer[l].blocks[b];

            nn_layernorm(nn, x, xn, blk->ln1w, blk->ln1b, L, C, 1e-5f);
            ss2d(m, blk, xn, m->ss);
            nn_mul_last(nn, x, blk->skip1, next, C, L);
            nn_lincomb(nn, next, m->ss, next, 1.0f, 1.0f, 0.0f);

            nn_layernorm(nn, next, xn, blk->ln2w, blk->ln2b, L, C, 1e-5f);
            nn_tokens2nchw(nn, xn, m->body, C, L);
            cab(m, blk, m->body, m->cab_out);
            nn_nchw2tokens(nn, m->cab_out, m->ss, C, L);
            nn_mul_last(nn, next, blk->skip2, x, C, L);
            nn_lincomb(nn, x, m->ss, x, 1.0f, 1.0f, 0.0f);
        }
        nn_tokens2nchw(nn, x, m->body, C, L);
        nn_conv(nn, m->body, m->body2, m->layer[l].conv_w,
                m->layer[l].conv_b, 1, ACT_NONE, 1, NULL);
        nn_nchw2tokens(nn, m->body2, x, C, L);
        nn_lincomb(nn, x, grp, x, 1.0f, 1.0f, 0.0f);
    }
    nn_layernorm(nn, x, xn, m->norm_w, m->norm_b, L, C, 1e-5f);
    nn_tokens2nchw(nn, xn, m->body, C, L);
    nn_conv(nn, m->body, m->body2, m->conv_body_w, m->conv_body_b,
            1, ACT_NONE, 1, NULL);
    nn_lincomb(nn, m->body2, m->first, m->body2, 1.0f, 1.0f, 0.0f);
    nn_conv(nn, m->body2, m->pre_up, m->conv_before_w, m->conv_before_b,
            1, ACT_LEAKY, 1, NULL);
    nn_conv(nn, m->pre_up, m->up1, m->up0_w, m->up0_b, 1, ACT_NONE, 2, NULL);
    nn_conv(nn, m->up1, m->up2, m->up2_w, m->up2_b, 1, ACT_NONE, 2, NULL);
    nn_conv(nn, m->up2, out, m->conv_last_w, m->conv_last_b,
            1, ACT_NONE, 1, NULL);
}
