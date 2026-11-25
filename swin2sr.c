#include "swin2sr.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void free_blk(struct swin_block *b)
{
    tensor_free(b->n1w); tensor_free(b->n1b);
    tensor_free(b->qkv_w); tensor_free(b->qkv_b);
    tensor_free(b->rel); tensor_free(b->logit);
    tensor_free(b->proj_w); tensor_free(b->proj_b);
    tensor_free(b->mask);
    tensor_free(b->n2w); tensor_free(b->n2b);
    tensor_free(b->fc1w); tensor_free(b->fc1b);
    tensor_free(b->fc2w); tensor_free(b->fc2b);
}

/* Continuous relative-position bias from Swin2SR's cpb_mlp, evaluated on
 * the host once per block (the table has at most 225 rows). */
static struct tensor *build_rel(struct ocl_backend *ocl,
                                struct tensor *table, struct tensor *cpb1w,
                                struct tensor *cpb1b, struct tensor *cpb2w,
                                int ws, int heads)
{
    int M = (2 * ws - 1) * (2 * ws - 1), N = ws * ws;
    float *tab, *w1, *b1, *w2, *h1, *out;
    struct tensor *t;
    int m, j, h, yq, xq, yk, xk, i, k;

    tab = malloc((size_t)M * 2 * sizeof(float));
    h1 = malloc((size_t)M * 512 * sizeof(float));
    out = malloc((size_t)heads * N * N * sizeof(float));
    w1 = malloc(512 * 2 * sizeof(float));
    b1 = malloc(512 * sizeof(float));
    w2 = malloc((size_t)heads * 512 * sizeof(float));
    if (!tab || !h1 || !out || !w1 || !b1 || !w2)
        die("out of memory");
    tensor_read(ocl, table, tab);
    tensor_read(ocl, cpb1w, w1);
    tensor_read(ocl, cpb1b, b1);
    tensor_read(ocl, cpb2w, w2);

    for (m = 0; m < M; m++) {
        for (j = 0; j < 512; j++) {
            float s = b1[j];
            s += tab[2 * m] * w1[2 * j] + tab[2 * m + 1] * w1[2 * j + 1];
            h1[(size_t)m * 512 + j] = s > 0.0f ? s : 0.0f;
        }
    }
    for (yq = 0; yq < ws; yq++) {
        for (xq = 0; xq < ws; xq++) {
            i = yq * ws + xq;
            for (yk = 0; yk < ws; yk++) {
                for (xk = 0; xk < ws; xk++) {
                    k = yk * ws + xk;
                    m = (2 * ws - 1) * (yq - yk + ws - 1) +
                        (xq - xk + ws - 1);
                    for (h = 0; h < heads; h++) {
                        float s = 0.0f;
                        for (j = 0; j < 512; j++)
                            s += h1[(size_t)m * 512 + j] *
                                 w2[(size_t)h * 512 + j];
                        s = 16.0f / (1.0f + expf(-s));
                        out[((size_t)h * N + i) * N + k] = s;
                    }
                }
            }
        }
    }
    t = tensor_upload(ocl, out, 1, heads, N, N);
    free(tab); free(h1); free(out); free(w1); free(b1); free(w2);
    return t;
}

static void free_scratch(struct swin2sr *m)
{
    int i;
    for (i = 0; i < 2; i++) {
        ocl_release(&m->maps[i].in);
        ocl_release(&m->maps[i].out);
    }
    for (i = 0; i < 3; i++) tensor_free(m->tok[i]);
    tensor_free(m->win); tensor_free(m->qkv);
    tensor_free(m->q); tensor_free(m->k); tensor_free(m->v);
    tensor_free(m->attn); tensor_free(m->ctx); tensor_free(m->proj);
    tensor_free(m->ffn1); tensor_free(m->ffn2);
    tensor_free(m->feat); tensor_free(m->feat2); tensor_free(m->feat64);
}

static void make_maps(struct swin2sr *m, int shift)
{
    struct ocl_backend *ocl = m->nn->ocl;
    int H = m->cur_H, W = m->cur_W, ws = m->cfg.window;
    int C = m->cfg.C, L = H * W;
    int nWh = H / ws, nWw = W / ws, nW = nWh * nWw, N = ws * ws;
    int *a = malloc((size_t)L * C * sizeof(int));
    int *b = malloc((size_t)L * C * sizeof(int));
    int wy, wx, iy, ix, t, c;

    if (!a || !b)
        die("out of memory");
    for (wy = 0; wy < nWh; wy++) {
        for (wx = 0; wx < nWw; wx++) {
            int w = wy * nWw + wx;
            for (iy = 0; iy < ws; iy++) {
                for (ix = 0; ix < ws; ix++) {
                    int sy = wy * ws + iy, sx = wx * ws + ix;
                    int dy = (sy + shift) % H, dx = (sx + shift) % W;
                    int src = dy * W + dx;
                    t = (w * N + iy * ws + ix);
                    for (c = 0; c < C; c++) {
                        a[(size_t)t * C + c] = src * C + c;
                        b[(size_t)src * C + c] = t * C + c;
                    }
                }
            }
        }
    }
    m->maps[shift ? 1 : 0].in =
        ocl_upload(ocl, a, (size_t)L * C * sizeof(int), "swin window map");
    m->maps[shift ? 1 : 0].out =
        ocl_upload(ocl, b, (size_t)L * C * sizeof(int), "swin reverse map");
    (void)nW;
    free(a);
    free(b);
}

static void make_mask(struct swin2sr *m, struct swin_block *blk, int shift)
{
    struct ocl_backend *ocl = m->nn->ocl;
    int H = m->cur_H, W = m->cur_W, ws = m->cfg.window;
    int nWh = H / ws, nWw = W / ws, nW = nWh * nWw, N = ws * ws;
    int *lab = malloc((size_t)H * W * sizeof(int));
    float *mask = calloc((size_t)nW * N * N, sizeof(float));
    int h, w, i, j;

    if (!lab || !mask)
        die("out of memory");
    /* Same three slices as SwinTransformerBlock.calculate_mask(). */
    for (h = 0; h < H; h++)
        for (w = 0; w < W; w++) {
            int sh = h < H - ws ? 0 : (h < H - shift ? 1 : 2);
            int sw = w < W - ws ? 0 : (w < W - shift ? 1 : 2);
            lab[h * W + w] = sh * 3 + sw;
        }
    for (h = 0; h < nWh; h++)
        for (w = 0; w < nWw; w++) {
            long base = (long)(h * nWw + w) * N * N;
            for (i = 0; i < N; i++)
                for (j = 0; j < N; j++) {
                    int a = lab[(h * ws + i / ws) * W + w * ws + i % ws];
                    int bb = lab[(h * ws + j / ws) * W + w * ws + j % ws];
                    mask[base + (long)i * N + j] = a != bb ? -100.0f : 0.0f;
                }
        }
    tensor_free(blk->mask);
    blk->mask = tensor_upload(ocl, mask, 1, nW, N, N);
    free(lab); free(mask);
}

static void prepare(struct swin2sr *m, int H, int W)
{
    struct ocl_backend *ocl = m->nn->ocl;
    int C = m->cfg.C, L = H * W, nW = (H / m->cfg.window) * (W / m->cfg.window);
    int N = m->N, d = m->d, heads = m->cfg.heads;
    int l, b;

    if (m->cur_H == H && m->cur_W == W)
        return;
    free_scratch(m);
    m->cur_H = H;
    m->cur_W = W;
    m->nW = nW;
    make_maps(m, 0);
    make_maps(m, m->cfg.window / 2);
    for (l = 0; l < m->cfg.layers; l++)
        for (b = 0; b < m->cfg.depth; b++) {
            struct swin_block *blk = &m->layer[l].blocks[b];
            if (blk->shifted)
                make_mask(m, blk, m->cfg.window / 2);
            else {
                tensor_free(blk->mask);
                blk->mask = tensor_alloc(ocl, 1, nW, N, N);
                tensor_zero(ocl, blk->mask);
            }
        }
    for (l = 0; l < 3; l++)
        m->tok[l] = tensor_alloc(ocl, 1, L, C, 1);
    m->win = tensor_alloc(ocl, 1, nW * N, C, 1);
    m->qkv = tensor_alloc(ocl, 1, nW * N, 3 * C, 1);
    m->q = tensor_alloc(ocl, 1, nW * heads, N * d, 1);
    m->k = tensor_alloc(ocl, 1, nW * heads, N * d, 1);
    m->v = tensor_alloc(ocl, 1, nW * heads, N * d, 1);
    m->attn = tensor_alloc(ocl, 1, nW * heads, N, N);
    m->ctx = tensor_alloc(ocl, 1, nW * heads, N * d, 1);
    m->proj = tensor_alloc(ocl, 1, nW * N, C, 1);
    m->ffn1 = tensor_alloc(ocl, 1, L, m->cfg.mlp_ratio * C, 1);
    m->ffn2 = tensor_alloc(ocl, 1, L, C, 1);
    m->feat = tensor_alloc(ocl, 1, C, H, W);
    m->feat2 = tensor_alloc(ocl, 1, C, H, W);
    m->feat64 = tensor_alloc(ocl, 1, 64, H, W);
}

void swin2sr_load(struct swin2sr *m, struct nn *nn, const char *file,
                  struct swin_cfg cfg)
{
    struct ocl_backend *ocl = nn->ocl;
    struct st_file f;
    long s4[4], s2[2], s1[1], s3[3];
    int l, b;
    char fmt[256];

    memset(m, 0, sizeof(*m));
    m->nn = nn;
    m->cfg = cfg;
    m->N = cfg.window * cfg.window;
    m->d = cfg.C / cfg.heads;
    if (m->d * cfg.heads != cfg.C)
        die("Swin2SR: embed_dim must be divisible by num_heads");
    st_open(&f, file);

#define LOAD1(field, name) do { \
        s1[0] = -1; (field) = tensor_load(ocl, &f, name, 1, s1); \
    } while (0)
#define LOAD2(field, name) do { \
        s2[0] = -1; s2[1] = -1; (field) = tensor_load(ocl, &f, name, 2, s2); \
    } while (0)
#define LOAD4(field, name) do { \
        s4[0] = -1; s4[1] = -1; s4[2] = -1; s4[3] = -1; \
        (field) = tensor_load(ocl, &f, name, 4, s4); \
    } while (0)

    LOAD4(m->conv_first_w, "conv_first.weight");
    LOAD1(m->conv_first_b, "conv_first.bias");
    LOAD4(m->pe_proj_w, "patch_embed.proj.weight");
    LOAD1(m->pe_proj_b, "patch_embed.proj.bias");
    LOAD1(m->pe_norm_w, "patch_embed.norm.weight");
    LOAD1(m->pe_norm_b, "patch_embed.norm.bias");
    LOAD1(m->norm_w, "norm.weight");
    LOAD1(m->norm_b, "norm.bias");
    LOAD4(m->conv_body_w, "conv_after_body.weight");
    LOAD1(m->conv_body_b, "conv_after_body.bias");
    LOAD4(m->conv_before_w, "conv_before_upsample.0.weight");
    LOAD1(m->conv_before_b, "conv_before_upsample.0.bias");
    LOAD4(m->conv_last_w, "conv_last.weight");
    LOAD1(m->conv_last_b, "conv_last.bias");

    m->layer = calloc(cfg.layers, sizeof(*m->layer));
    if (!m->layer)
        die("out of memory");
    for (l = 0; l < cfg.layers; l++) {
        struct swin_layer *lay = &m->layer[l];
        lay->blocks = calloc(cfg.depth, sizeof(*lay->blocks));
        if (!lay->blocks)
            die("out of memory");
        snprintf(fmt, sizeof(fmt), "layers.%d.conv.weight", l);
        LOAD4(lay->conv_w, fmt);
        snprintf(fmt, sizeof(fmt), "layers.%d.conv.bias", l);
        LOAD1(lay->conv_b, fmt);
        for (b = 0; b < cfg.depth; b++) {
            struct swin_block *blk = &lay->blocks[b];
            struct tensor *tab, *c1w, *c1b, *c2w, *logit;
            float *lv, *scale;
            int h;

            blk->shifted = b % 2;
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.norm1.weight", l, b);
            LOAD1(blk->n1w, fmt);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.norm1.bias", l, b);
            LOAD1(blk->n1b, fmt);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.attn.qkv.weight", l, b);
            LOAD2(blk->qkv_w, fmt);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.attn.q_bias", l, b);
            s1[0] = cfg.C; tab = tensor_load(ocl, &f, fmt, 1, s1);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.attn.v_bias", l, b);
            s1[0] = cfg.C; c1b = tensor_load(ocl, &f, fmt, 1, s1);
            lv = calloc(3 * cfg.C, sizeof(float));
            scale = malloc(cfg.C * sizeof(float));
            if (!lv || !scale)
                die("out of memory");
            tensor_read(ocl, tab, lv);
            tensor_read(ocl, c1b, lv + 2 * cfg.C);
            tensor_free(tab); tensor_free(c1b);
            blk->qkv_b = tensor_upload(ocl, lv, 1, 3 * cfg.C, 1, 1);

            s4[0] = 1; s4[1] = 2 * cfg.window - 1;
            s4[2] = 2 * cfg.window - 1; s4[3] = 2;
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.attn.relative_coords_table", l, b);
            tab = tensor_load(ocl, &f, fmt, 4, s4);
            s2[0] = 512; s2[1] = 2;
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.attn.cpb_mlp.0.weight", l, b);
            c1w = tensor_load(ocl, &f, fmt, 2, s2);
            s1[0] = 512;
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.attn.cpb_mlp.0.bias", l, b);
            c1b = tensor_load(ocl, &f, fmt, 1, s1);
            s2[0] = cfg.heads; s2[1] = 512;
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.attn.cpb_mlp.2.weight", l, b);
            c2w = tensor_load(ocl, &f, fmt, 2, s2);
            blk->rel = build_rel(ocl, tab, c1w, c1b, c2w, cfg.window, cfg.heads);
            tensor_free(tab); tensor_free(c1w); tensor_free(c1b); tensor_free(c2w);

            s3[0] = cfg.heads; s3[1] = 1; s3[2] = 1;
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.attn.logit_scale", l, b);
            logit = tensor_load(ocl, &f, fmt, 3, s3);
            tensor_read(ocl, logit, scale);
            for (h = 0; h < cfg.heads; h++) {
                float v = scale[h];
                if (v > 4.605170186f) v = 4.605170186f;
                scale[h] = expf(v);
            }
            tensor_free(logit);
            blk->logit = tensor_upload(ocl, scale, 1, cfg.heads, 1, 1);
            free(lv); free(scale);

            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.attn.proj.weight", l, b);
            LOAD2(blk->proj_w, fmt);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.attn.proj.bias", l, b);
            LOAD1(blk->proj_b, fmt);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.norm2.weight", l, b);
            LOAD1(blk->n2w, fmt);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.norm2.bias", l, b);
            LOAD1(blk->n2b, fmt);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.mlp.fc1.weight", l, b);
            s2[0] = cfg.mlp_ratio * cfg.C; s2[1] = cfg.C;
            blk->fc1w = tensor_load(ocl, &f, fmt, 2, s2);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.mlp.fc1.bias", l, b);
            s1[0] = cfg.mlp_ratio * cfg.C;
            blk->fc1b = tensor_load(ocl, &f, fmt, 1, s1);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.mlp.fc2.weight", l, b);
            s2[0] = cfg.C; s2[1] = cfg.mlp_ratio * cfg.C;
            blk->fc2w = tensor_load(ocl, &f, fmt, 2, s2);
            snprintf(fmt, sizeof(fmt), "layers.%d.residual_group.blocks.%d.mlp.fc2.bias", l, b);
            s1[0] = cfg.C;
            blk->fc2b = tensor_load(ocl, &f, fmt, 1, s1);
        }
    }
#undef LOAD1
#undef LOAD2
#undef LOAD4
    st_close(&f);
}

void swin2sr_free(struct swin2sr *m)
{
    int l, b;
    free_scratch(m);
    tensor_free(m->conv_first_w); tensor_free(m->conv_first_b);
    tensor_free(m->pe_proj_w); tensor_free(m->pe_proj_b);
    tensor_free(m->pe_norm_w); tensor_free(m->pe_norm_b);
    tensor_free(m->norm_w); tensor_free(m->norm_b);
    tensor_free(m->conv_body_w); tensor_free(m->conv_body_b);
    tensor_free(m->conv_before_w); tensor_free(m->conv_before_b);
    tensor_free(m->conv_last_w); tensor_free(m->conv_last_b);
    if (m->layer) {
        for (l = 0; l < m->cfg.layers; l++) {
            tensor_free(m->layer[l].conv_w);
            tensor_free(m->layer[l].conv_b);
            for (b = 0; b < m->cfg.depth; b++)
                free_blk(&m->layer[l].blocks[b]);
            free(m->layer[l].blocks);
        }
        free(m->layer);
    }
}

void swin2sr_forward(struct swin2sr *m, struct tensor *in, struct tensor *out)
{
    struct nn *nn = m->nn;
    int H = in->h, W = in->w, C = m->cfg.C, L = H * W;
    int N = m->N, nW, heads = m->cfg.heads, d = m->d;
    int l, b, map;
    struct tensor *x, *xn, *next;

    if (H % m->cfg.window || W % m->cfg.window)
        die("Swin2SR: input size must be a multiple of window_size");
    prepare(m, H, W);
    nW = m->nW;

    /* Shallow features: conv_first, patch projection and patch norm. */
    nn_conv(nn, in, m->feat, m->conv_first_w, m->conv_first_b, 1, ACT_NONE, 1, NULL);
    nn_conv(nn, m->feat, m->feat2, m->pe_proj_w, m->pe_proj_b, 1, ACT_NONE, 1, NULL);
    nn_nchw2tokens(nn, m->feat2, m->tok[0], C, L);
    nn_layernorm(nn, m->tok[0], m->tok[1], m->pe_norm_w, m->pe_norm_b, L, C, 1e-5f);
    x = m->tok[1];
    xn = m->tok[0];
    next = m->tok[2];

    /* Transformer body. */
    for (l = 0; l < m->cfg.layers; l++) {
        struct swin_layer *lay = &m->layer[l];
        for (b = 0; b < m->cfg.depth; b++) {
            struct swin_block *blk = &lay->blocks[b];
            map = blk->shifted ? 1 : 0;

            nn_layernorm(nn, x, xn, blk->n1w, blk->n1b, L, C, 1e-5f);
            nn_gather(nn, xn, m->win, m->maps[map].in, L * C);
            nn_gemm(nn, m->win, blk->qkv_w, blk->qkv_b, m->qkv,
                    nW * N, C, 3 * C);
            nn_qkv_split(nn, m->qkv, m->q, m->k, m->v, nW, heads, N, C);
            nn_attn_qk(nn, m->q, m->k, blk->logit, m->attn, nW, heads, N, d);
            nn_attn_bias(nn, m->attn, blk->rel, blk->mask, nW, heads, N,
                         nW * heads * N * N);
            nn_softmax(nn, m->attn, m->attn, nW * heads, N);
            nn_attn_av(nn, m->attn, m->v, m->ctx, nW, heads, N, d);
            nn_head_concat(nn, m->ctx, m->proj, nW, heads, N, C);
            nn_gemm(nn, m->proj, blk->proj_w, blk->proj_b, m->ffn2,
                    nW * N, C, C);
            nn_gather(nn, m->ffn2, m->ffn1, m->maps[map].out, L * C);
            nn_lincomb(nn, x, m->ffn1, next, 1.0f, 1.0f, 0.0f);

            nn_layernorm(nn, next, xn, blk->n2w, blk->n2b, L, C, 1e-5f);
            nn_gemm(nn, xn, blk->fc1w, blk->fc1b, m->ffn1, L, C,
                    m->cfg.mlp_ratio * C);
            nn_act(nn, m->ffn1, m->ffn1, ACT_GELU, 0.0f);
            nn_gemm(nn, m->ffn1, blk->fc2w, blk->fc2b, m->ffn2,
                    L, m->cfg.mlp_ratio * C, C);
            nn_lincomb(nn, next, m->ffn2, x, 1.0f, 1.0f, 0.0f);
        }
        nn_tokens2nchw(nn, x, m->feat, C, L);
        nn_conv(nn, m->feat, m->feat2, lay->conv_w, lay->conv_b,
                1, ACT_NONE, 1, NULL);
        nn_nchw2tokens(nn, m->feat2, x, C, L);
    }
    nn_layernorm(nn, x, xn, m->norm_w, m->norm_b, L, C, 1e-5f);
    nn_tokens2nchw(nn, xn, m->feat, C, L);
    nn_conv(nn, m->feat, m->feat2, m->conv_body_w, m->conv_body_b,
            1, ACT_NONE, 1, NULL);
    nn_lincomb(nn, m->feat2, m->feat, m->feat2, 1.0f, 1.0f, 0.0f);
    nn_conv(nn, m->feat2, m->feat64, m->conv_before_w, m->conv_before_b,
            1, ACT_LEAKY, 1, NULL);
    nn_conv(nn, m->feat64, out, m->conv_last_w, m->conv_last_b,
            1, ACT_NONE, 1, NULL);
}
