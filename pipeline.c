#include "pipeline.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PATH_MAX_LEN 1024

static void path_join(char *dst, size_t n, const char *dir, const char *file)
{
    snprintf(dst, n, "%s/%s", dir, file);
}

static cl_mem spec_alloc(struct ocl_backend *ocl, int C, int sz)
{
    return ocl_alloc(ocl, (size_t)C * sz * sz * sizeof(cl_float2), "hc spectrum");
}

/* Reorder ten channels with an explicit index map. */
static void reorder(struct nn *nn, struct tensor *in, struct tensor *out,
                    const int *idx, int K, int HW)
{
    nn_gather_bands(nn, in, out, idx, K, HW);
}

static void reconstruct(struct nn *nn, struct tensor *a, struct tensor *b,
                        struct tensor *out, int H, int W)
{
    static const int idx[10] = {0, 1, 2, 4, 5, 6, 3, 7, 8, 9};
    struct tensor *tmp = tensor_alloc(nn->ocl, 1, 10, H, W);

    nn_concat2(nn, a, b, tmp, 4, 6, H * W);
    reorder(nn, tmp, out, idx, 10, H * W);
    tensor_free(tmp);
}

void pipeline_load(struct pipeline *p, struct nn *nn, const char *model_dir)
{
    struct ocl_backend *ocl = nn->ocl;
    char path[PATH_MAX_LEN];
    struct swin_cfg f2_cfg = {128, 128, 192, 8, 4, 8, 8, 4, 6};
    struct swin_cfg f4_cfg = {128, 128, 120, 8, 8, 8, 8, 4, 6};
    struct mamba_cfg m_cfg = {128, 128, 4, 4, 96, 384, 6, 8, 4, 6, 16};

    memset(p, 0, sizeof(*p));
    p->nn = nn;

    path_join(path, sizeof(path), model_dir, "sr_model.safetensor");
    mamba_sr_load(&p->rgbn, nn, path, m_cfg);
    path_join(path, sizeof(path), model_dir, "f2_model.safetensor");
    swin2sr_load(&p->f2, nn, path, f2_cfg);
    path_join(path, sizeof(path), model_dir, "model.safetensor");
    swin2sr_load(&p->f4, nn, path, f4_cfg);

    resize_map_build(ocl, &p->bicubic_same, 128, 128, 1);
    resize_map_build(ocl, &p->bicubic_x4, 128, 512, 1);
    resize_map_build(ocl, &p->bilinear_x2, 64, 128, 0);
    resize_map_build(ocl, &p->bilinear_x4, 128, 512, 0);

    path_join(path, sizeof(path), model_dir, "sr_hard_constraint.safetensor");
    p->hc_rgbn = hc_mask_load(ocl, path, 512);
    p->spec_rgbn = spec_alloc(ocl, 4, 512);
    path_join(path, sizeof(path), model_dir, "f2_hard_constraint.safetensor");
    p->hc_f2 = hc_mask_load(ocl, path, 128);
    p->spec_f2 = spec_alloc(ocl, 6, 128);
    path_join(path, sizeof(path), model_dir, "hard_constraint.safetensor");
    p->hc_f4 = hc_mask_load(ocl, path, 512);
    p->spec_f4 = spec_alloc(ocl, 6, 512);
}

void pipeline_free(struct pipeline *p)
{
    mamba_sr_free(&p->rgbn);
    swin2sr_free(&p->f2);
    swin2sr_free(&p->f4);
    resize_map_free(&p->bicubic_same);
    resize_map_free(&p->bicubic_x4);
    resize_map_free(&p->bilinear_x2);
    resize_map_free(&p->bilinear_x4);
    ocl_release(&p->hc_rgbn);
    ocl_release(&p->hc_f2);
    ocl_release(&p->hc_f4);
    ocl_release(&p->spec_rgbn);
    ocl_release(&p->spec_f2);
    ocl_release(&p->spec_f4);
}

void pipeline_forward(struct pipeline *p, struct tensor *in, struct tensor *out)
{
    struct nn *nn = p->nn;
    struct ocl_backend *ocl = nn->ocl;
    static const int b20[6] = {3, 4, 5, 7, 8, 9};
    static const int b10[4] = {0, 1, 2, 6};
    static const int brgbn[4] = {2, 1, 0, 6};
    static const int breorder[4] = {2, 1, 0, 3};
    int H = in->h, W = in->w, HW = H * W;
    int OH = 4 * H, OW = 4 * W, OHW = OH * OW;
    struct tensor *bands20, *down, *up, *native, *resampled;
    struct tensor *f2out, *f2hc, *all10;
    struct tensor *rgbn_in, *rgbn_sr, *rgbn_hc, *rgbn_re;
    struct tensor *swir10, *swir_up, *fusion, *f4out, *swir;
    struct tensor *tmp_x2, *tmp_x4_4, *tmp_x4_6;

    if (H != 128 || W != 128)
        die("smoke test expects a 128x128 input");
    if (out->n != 1 || out->c != 10 || out->h != OH || out->w != OW)
        die("output tensor has the wrong shape");

    bands20 = tensor_alloc(ocl, 1, 6, H, W);
    down = tensor_alloc(ocl, 1, 6, H / 2, W / 2);
    up = tensor_alloc(ocl, 1, 6, H, W);
    native = tensor_alloc(ocl, 1, 4, H, W);
    resampled = tensor_alloc(ocl, 1, 10, H, W);
    f2out = tensor_alloc(ocl, 1, 6, H, W);
    f2hc = tensor_alloc(ocl, 1, 6, H, W);
    all10 = tensor_alloc(ocl, 1, 10, H, W);
    rgbn_in = tensor_alloc(ocl, 1, 4, H, W);
    rgbn_sr = tensor_alloc(ocl, 1, 4, OH, OW);
    rgbn_hc = tensor_alloc(ocl, 1, 4, OH, OW);
    rgbn_re = tensor_alloc(ocl, 1, 4, OH, OW);
    swir10 = tensor_alloc(ocl, 1, 6, H, W);
    swir_up = tensor_alloc(ocl, 1, 6, OH, OW);
    fusion = tensor_alloc(ocl, 1, 10, OH, OW);
    f4out = tensor_alloc(ocl, 1, 6, OH, OW);
    swir = tensor_alloc(ocl, 1, 6, OH, OW);
    tmp_x2 = tensor_alloc(ocl, 1, 6, H / 2, OW / 4);
    tmp_x4_4 = tensor_alloc(ocl, 1, 4, H, OW);
    tmp_x4_6 = tensor_alloc(ocl, 1, 6, H, OW);

    /* Stage 1: referencex2 / f2 model at 10 m. */
    reorder(nn, in, bands20, b20, 6, HW);
    reorder(nn, in, native, b10, 4, HW);
    nn_decimate2(nn, bands20, down, H, W);
    nn_resize(nn, &p->bilinear_x2, down, tmp_x2, up, 6);
    nn_concat2(nn, up, native, resampled, 6, 4, HW);
    swin2sr_forward(&p->f2, resampled, f2out);
    nn_clamp(nn, f2out);
    nn_resize(nn, &p->bicubic_same, resampled, bands20, f2hc, 6);
    nn_hc(nn, f2hc, f2out, p->spec_f2, p->hc_f2, H, 6);
    reconstruct(nn, native, f2out, all10, H, W);

    /* Stage 2: MambaSR RGBN at 2.5 m. */
    reorder(nn, in, rgbn_in, brgbn, 4, HW);
    mamba_sr_forward(&p->rgbn, rgbn_in, rgbn_sr);
    nn_clamp(nn, rgbn_sr);
    nn_resize(nn, &p->bicubic_x4, rgbn_in, tmp_x4_4, rgbn_hc, 4);
    nn_hc(nn, rgbn_hc, rgbn_sr, p->spec_rgbn, p->hc_rgbn, OH, 4);
    reorder(nn, rgbn_sr, rgbn_re, breorder, 4, OHW);

    /* Stage 3: Swin2SR fusion at 2.5 m and final hard constraint. */
    reorder(nn, all10, swir10, b20, 6, HW);
    nn_resize(nn, &p->bilinear_x4, swir10, tmp_x4_6, swir_up, 6);
    nn_concat2(nn, swir_up, rgbn_re, fusion, 6, 4, OHW);
    swin2sr_forward(&p->f4, fusion, f4out);
    nn_clamp(nn, f4out);
    nn_resize(nn, &p->bicubic_x4, swir10, tmp_x4_6, swir, 6);
    nn_hc(nn, swir, f4out, p->spec_f4, p->hc_f4, OH, 6);
    reconstruct(nn, rgbn_re, f4out, out, OH, OW);

    tensor_free(bands20); tensor_free(down); tensor_free(up);
    tensor_free(native); tensor_free(resampled); tensor_free(f2out);
    tensor_free(f2hc); tensor_free(all10); tensor_free(rgbn_in);
    tensor_free(rgbn_sr); tensor_free(rgbn_hc); tensor_free(rgbn_re);
    tensor_free(swir10); tensor_free(swir_up); tensor_free(fusion);
    tensor_free(f4out); tensor_free(swir);
    tensor_free(tmp_x2); tensor_free(tmp_x4_4); tensor_free(tmp_x4_6);
}
