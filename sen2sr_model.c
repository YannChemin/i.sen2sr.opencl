/****************************************************************************
 *
 * MODULE:       i.sen2sr.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      SEN2SRLite super-resolution networks and their
 *               radiometric hard constraint on an OpenCL device.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

/* This is a port of the inference path of the sen2sr Python package
 * (https://github.com/ESAOpenSR/sen2sr) for its CNN "Lite" models:
 *
 * - CNNSR (models/opensr_baseline/cnn.py) in evaluation mode, where each
 *   Conv3XC (1x1 -> 3x3 -> 1x1 convolutions plus a 1x1 skip) is folded
 *   into a single 3x3 convolution, as Conv3XC.update_params() does.
 *   CNNSR.forward() feeds the same feature map to every SPAB block and
 *   only keeps the outputs of the first and the last one, so the blocks
 *   in between do not change the result and are not evaluated. SPAB's
 *   in-place SiLU also means that its second output is silu(c1_r(x)).
 * - HardConstraint (models/tricks.py): the low frequencies of the output
 *   are replaced by those of the bicubic-upsampled input, which by
 *   linearity of the FFT equals sr + Re(ifft2(M * fft2(lr_up - sr))),
 *   with M the stored mask moved back to unshifted frequency order.
 * - The nonreference, referencex2 and referencex4 wrappers that chain
 *   these parts for the three released model variants. */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "safetensors.h"
#include "sen2sr_model.h"

#define ACT_NONE 0
#define ACT_SILU 1
#define ACT_ATTN 2
#define ACT_RELU 3

#define COB 4 /* Output channels per work-item in conv2d. */

const char *sr_variant_name(enum sr_variant v)
{
    switch (v) {
    case SR_RGBN_X4:
        return "SEN2SRLite NonReference_RGBN_x4";
    case SR_RSWIR_X2:
        return "SEN2SRLite Reference_RSWIR_x2";
    default:
        return "SEN2SRLite main (all bands x4)";
    }
}

/* Model loading. */

static struct sr_conv upload_conv(struct ocl_backend *ocl, const float *w,
                                  const float *b, int cin, int cout, int k,
                                  const char *what)
{
    struct sr_conv c;

    c.cin = cin;
    c.cout = cout;
    c.k = k;
    c.w = ocl_upload(ocl, w, sizeof(float) * cout * cin * k * k, what);
    c.b = ocl_upload(ocl, b, sizeof(float) * cout, what);
    return c;
}

static struct sr_conv load_plain_conv(struct ocl_backend *ocl,
                                      const struct st_file *f,
                                      const char *prefix, int cin, int cout,
                                      int k)
{
    char name[256];
    long ws[4] = {cout, cin, k, k}, bs[1] = {cout};
    float *w, *b;
    struct sr_conv c;

    snprintf(name, sizeof(name), "%s.weight", prefix);
    w = st_get_f32(f, name, 4, ws);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    b = st_get_f32(f, name, 1, bs);
    c = upload_conv(ocl, w, b, cin, cout, k, prefix);
    G_free(w);
    G_free(b);
    return c;
}

/* Fold a Conv3XC into one 3x3 convolution: the 1x1 (w1), 3x3 (w2) and
 * 1x1 (w3) chain composes into W = w3 * w2 * w1, the biases propagate
 * through the later layers, and the 1x1 skip adds to the kernel
 * centre. */
static struct sr_conv load_conv3xc(struct ocl_backend *ocl,
                                   const struct st_file *f, const char *prefix,
                                   int cin, int cout)
{
    char name[256];
    long s1[4] = {-1, cin, 1, 1}, s2[4] = {-1, -1, 3, 3};
    long s3[4] = {cout, -1, 1, 1}, ssk[4] = {cout, cin, 1, 1}, sb[1];
    float *w1, *b1, *w2, *b2, *w3, *b3, *sk, *skb, *w, *b;
    double *t;
    int m1, m2, o, i, p, a, c;
    struct sr_conv conv;

    snprintf(name, sizeof(name), "%s.conv.0.weight", prefix);
    w1 = st_get_f32(f, name, 4, s1);
    m1 = s1[0];
    s2[1] = m1;
    snprintf(name, sizeof(name), "%s.conv.1.weight", prefix);
    w2 = st_get_f32(f, name, 4, s2);
    m2 = s2[0];
    s3[1] = m2;
    snprintf(name, sizeof(name), "%s.conv.2.weight", prefix);
    w3 = st_get_f32(f, name, 4, s3);
    snprintf(name, sizeof(name), "%s.sk.weight", prefix);
    sk = st_get_f32(f, name, 4, ssk);

    sb[0] = m1;
    snprintf(name, sizeof(name), "%s.conv.0.bias", prefix);
    b1 = st_get_f32(f, name, 1, sb);
    sb[0] = m2;
    snprintf(name, sizeof(name), "%s.conv.1.bias", prefix);
    b2 = st_get_f32(f, name, 1, sb);
    sb[0] = cout;
    snprintf(name, sizeof(name), "%s.conv.2.bias", prefix);
    b3 = st_get_f32(f, name, 1, sb);
    snprintf(name, sizeof(name), "%s.sk.bias", prefix);
    skb = st_get_f32(f, name, 1, sb);

    /* t = w2 * w1, (m2, cin, 9). */
    t = G_calloc((size_t)m2 * cin * 9, sizeof(double));
    for (o = 0; o < m2; o++)
        for (a = 0; a < m1; a++)
            for (i = 0; i < cin; i++) {
                double v = w1[a * cin + i];

                for (p = 0; p < 9; p++)
                    t[(o * cin + i) * 9 + p] += v * w2[(o * m1 + a) * 9 + p];
            }

    w = G_malloc(sizeof(float) * cout * cin * 9);
    b = G_malloc(sizeof(float) * cout);
    for (o = 0; o < cout; o++) {
        double bias = (double)b3[o] + skb[o];

        for (i = 0; i < cin; i++)
            for (p = 0; p < 9; p++) {
                double v = 0.0;

                for (c = 0; c < m2; c++)
                    v += (double)w3[o * m2 + c] * t[(c * cin + i) * 9 + p];
                if (p == 4)
                    v += sk[o * cin + i];
                w[(o * cin + i) * 9 + p] = (float)v;
            }
        for (c = 0; c < m2; c++) {
            double v = b2[c];

            for (a = 0; a < m1; a++)
                for (p = 0; p < 9; p++)
                    v += (double)w2[(c * m1 + a) * 9 + p] * b1[a];
            bias += (double)w3[o * m2 + c] * v;
        }
        b[o] = (float)bias;
    }

    conv = upload_conv(ocl, w, b, cin, cout, 3, prefix);
    G_free(t);
    G_free(w);
    G_free(b);
    G_free(w1);
    G_free(b1);
    G_free(w2);
    G_free(b2);
    G_free(w3);
    G_free(b3);
    G_free(sk);
    G_free(skb);
    return conv;
}

static void load_cnn(struct ocl_backend *ocl, const char *path,
                     struct sr_cnn *net, int cin, int cout)
{
    struct st_file f;
    long shape[4] = {-1, cin, 1, 1};
    char prefix[64];
    float *tmp;
    int nblocks, up2, i;
    const char *sub[3] = {"c1_r", "c2_r", "c3_r"};

    st_open(&f, path);
    if (!st_find(&f, "conv_1.conv.0.weight"))
        G_fatal_error(_("<%s> does not hold a SEN2SRLite CNN; only the "
                        "SEN2SRLite models are supported"),
                      path);

    tmp = st_get_f32(&f, "conv_1.conv.0.weight", 4, shape);
    G_free(tmp);
    shape[0] = -1;
    shape[1] = -1;
    tmp = st_get_f32(&f, "conv_1.conv.2.weight", 4, shape);
    G_free(tmp);
    net->cin = cin;
    net->cout = cout;
    net->feat = shape[0];

    for (nblocks = 0;; nblocks++) {
        snprintf(prefix, sizeof(prefix), "blocks.%d.c1_r.conv.0.weight",
                 nblocks);
        if (!st_find(&f, prefix))
            break;
    }
    if (nblocks == 0)
        G_fatal_error(_("<%s> has no attention blocks"), path);

    shape[0] = -1;
    shape[1] = net->feat;
    shape[2] = 3;
    shape[3] = 3;
    tmp = st_get_f32(&f, "upsampler.0.weight", 4, shape);
    G_free(tmp);
    up2 = shape[0] / cout;
    for (net->up = 1; net->up * net->up < up2; net->up++)
        ;
    if (shape[0] % cout || net->up * net->up != up2)
        G_fatal_error(_("<%s>: upsampler of %ld channels does not match %d "
                        "output bands"),
                      path, shape[0], cout);

    G_debug(1, "%s: cin=%d cout=%d feat=%d blocks=%d up=%d", path, cin, cout,
            net->feat, nblocks, net->up);

    net->conv1 = load_conv3xc(ocl, &f, "conv_1", cin, net->feat);
    for (i = 0; i < 3; i++) {
        snprintf(prefix, sizeof(prefix), "blocks.0.%s", sub[i]);
        net->first[i] = load_conv3xc(ocl, &f, prefix, net->feat, net->feat);
        snprintf(prefix, sizeof(prefix), "blocks.%d.%s", nblocks - 1, sub[i]);
        net->last[i] = load_conv3xc(ocl, &f, prefix, net->feat, net->feat);
    }
    net->conv2 = load_conv3xc(ocl, &f, "conv_2", net->feat, net->feat);
    net->cat =
        load_plain_conv(ocl, &f, "conv_cat", 4 * net->feat, net->feat, 1);
    net->ups = load_plain_conv(ocl, &f, "upsampler.0", net->feat,
                               cout * net->up * net->up, 3);
    st_close(&f);
}

static void free_conv(struct sr_conv *c)
{
    ocl_release(&c->w);
    ocl_release(&c->b);
}

static void free_cnn(struct sr_cnn *net)
{
    int i;

    free_conv(&net->conv1);
    for (i = 0; i < 3; i++) {
        free_conv(&net->first[i]);
        free_conv(&net->last[i]);
    }
    free_conv(&net->conv2);
    free_conv(&net->cat);
    free_conv(&net->ups);
}

static void load_hc(struct ocl_backend *ocl, const char *path, struct sr_hc *hc,
                    int n)
{
    struct st_file f;
    long shape[2] = {n, n};
    float *mask, *shifted;
    int y, x, h = n / 2;

    st_open(&f, path);
    mask = st_get_f32(&f, "weights", 2, shape);
    st_close(&f);

    /* torch fftshift()s the spectrum before masking; moving the mask the
     * other way (ifftshift, the same for even n) keeps the spectrum in
     * natural order. */
    shifted = G_malloc(sizeof(float) * n * n);
    for (y = 0; y < n; y++)
        for (x = 0; x < n; x++)
            shifted[y * n + x] = mask[((y + h) % n) * n + (x + h) % n];
    hc->n = n;
    hc->mask = ocl_upload(ocl, shifted, sizeof(float) * n * n, path);
    G_free(mask);
    G_free(shifted);
}

/* Anti-aliased resampling weights of torch.nn.functional.interpolate(...,
 * antialias=True) for bilinear (cubic = 0) and bicubic (cubic = 1)
 * modes: a PIL-style filter (Keys cubic with a = -0.5) whose support
 * widens when downsampling, clipped at the borders and renormalised. */
static double aa_filter(double x, int cubic)
{
    const double a = -0.5;

    x = fabs(x);
    if (!cubic)
        return x < 1.0 ? 1.0 - x : 0.0;
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

static void build_resize(struct ocl_backend *ocl, struct sr_resize *r, int in,
                         int out, int cubic)
{
    double scale = (double)in / out;
    double support = (cubic ? 2.0 : 1.0) * (scale >= 1.0 ? scale : 1.0);
    double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    int *idx, i, j, kmax = 0;
    float *wts;

    idx = G_malloc(sizeof(int) * 2 * out);
    for (i = 0; i < out; i++) {
        double center = scale * (i + 0.5);
        int xmin = (int)(center - support + 0.5);
        int xmax = (int)(center + support + 0.5);

        if (xmin < 0)
            xmin = 0;
        if (xmax > in)
            xmax = in;
        idx[2 * i] = xmin;
        idx[2 * i + 1] = xmax - xmin;
        if (xmax - xmin > kmax)
            kmax = xmax - xmin;
    }
    wts = G_calloc((size_t)out * kmax, sizeof(float));
    for (i = 0; i < out; i++) {
        double center = scale * (i + 0.5), total = 0.0, w[64];
        int xmin = idx[2 * i], n = idx[2 * i + 1];

        if (n > 64)
            G_fatal_error(_("Resampling support too wide"));
        for (j = 0; j < n; j++) {
            w[j] = aa_filter((j + xmin - center + 0.5) * invscale, cubic);
            total += w[j];
        }
        for (j = 0; j < n; j++)
            wts[i * kmax + j] = total != 0.0 ? (float)(w[j] / total) : 0.0f;
    }
    r->in = in;
    r->out = out;
    r->kmax = kmax;
    r->idx = ocl_upload(ocl, idx, sizeof(int) * 2 * out, "resize");
    r->wts = ocl_upload(ocl, wts, sizeof(float) * out * kmax, "resize");
    G_free(idx);
    G_free(wts);
}

static void free_resize(struct sr_resize *r)
{
    ocl_release(&r->idx);
    ocl_release(&r->wts);
}

static int file_exists(const char *dir, const char *name, char *path,
                       size_t size)
{
    snprintf(path, size, "%s/%s", dir, name);
    return access(path, R_OK) == 0;
}

static void require_file(const char *dir, const char *name, char *path,
                         size_t size)
{
    if (!file_exists(dir, name, path, size))
        G_fatal_error(_("Model file <%s> not found or not readable"), path);
}

static void set_bands(enum s2_band *dst, const enum s2_band *src, int n)
{
    memcpy(dst, src, n * sizeof(*dst));
}

void sr_model_load(struct sr_model *m, struct ocl_backend *ocl, const char *dir)
{
    static const enum s2_band all[S2_NBANDS] = {B02, B03, B04, B05, B06,
                                                B07, B08, B8A, B11, B12};
    static const enum s2_band rgbn[4] = {B02, B03, B04, B08};
    static const enum s2_band rswir[6] = {B05, B06, B07, B8A, B11, B12};
    const int N = SR_TILE;
    char path[GPATH_MAX];
    size_t plane, hr_plane, big;
    int feat, max_side;

    memset(m, 0, sizeof(*m));
    m->ocl = ocl;

    if (file_exists(dir, "sr_model.safetensor", path, sizeof(path))) {
        m->variant = SR_ALL_X4;
        load_cnn(ocl, path, &m->rgbn, 4, 4);
        require_file(dir, "sr_hard_constraint.safetensor", path, sizeof(path));
        load_hc(ocl, path, &m->hc_rgbn, N * m->rgbn.up);
        require_file(dir, "f2_model.safetensor", path, sizeof(path));
        load_cnn(ocl, path, &m->f2, 10, 6);
        require_file(dir, "f2_hard_constraint.safetensor", path, sizeof(path));
        load_hc(ocl, path, &m->hc_f2, N);
        require_file(dir, "model.safetensor", path, sizeof(path));
        load_cnn(ocl, path, &m->fus, 10, 6);
        require_file(dir, "hard_constraint.safetensor", path, sizeof(path));
        load_hc(ocl, path, &m->hc_fus, N * m->rgbn.up);
        if (m->rgbn.up != 4 || m->f2.up != 1 || m->fus.up != 1)
            G_fatal_error(_("Unexpected upscaling factors in model <%s>"), dir);
        m->scale = 4;
        m->nin = m->nout = S2_NBANDS;
        set_bands(m->in_bands, all, S2_NBANDS);
        set_bands(m->out_bands, all, S2_NBANDS);
    }
    else {
        struct st_file f;
        long shape[4] = {-1, -1, 1, 1};
        float *tmp;

        require_file(dir, "model.safetensor", path, sizeof(path));
        st_open(&f, path);
        if (!st_find(&f, "conv_1.conv.0.weight"))
            G_fatal_error(_("<%s> does not hold a SEN2SRLite CNN; only the "
                            "SEN2SRLite models are supported"),
                          path);
        tmp = st_get_f32(&f, "conv_1.conv.0.weight", 4, shape);
        G_free(tmp);
        st_close(&f);

        if (shape[1] == 4) {
            m->variant = SR_RGBN_X4;
            load_cnn(ocl, path, &m->rgbn, 4, 4);
            if (m->rgbn.up != 4)
                G_fatal_error(_("Unexpected upscaling factor %d in <%s>"),
                              m->rgbn.up, path);
            require_file(dir, "hard_constraint.safetensor", path, sizeof(path));
            load_hc(ocl, path, &m->hc_rgbn, N * 4);
            m->scale = 4;
            m->nin = m->nout = 4;
            set_bands(m->in_bands, rgbn, 4);
            set_bands(m->out_bands, rgbn, 4);
        }
        else if (shape[1] == 10) {
            m->variant = SR_RSWIR_X2;
            load_cnn(ocl, path, &m->f2, 10, 6);
            if (m->f2.up != 1)
                G_fatal_error(_("Unexpected upscaling factor %d in <%s>"),
                              m->f2.up, path);
            require_file(dir, "hard_constraint.safetensor", path, sizeof(path));
            load_hc(ocl, path, &m->hc_f2, N);
            /* The 20 m bands are restored on the 10 m input grid. */
            m->scale = 1;
            m->nin = S2_NBANDS;
            m->nout = 6;
            set_bands(m->in_bands, all, S2_NBANDS);
            set_bands(m->out_bands, rswir, 6);
        }
        else
            G_fatal_error(_("<%s>: unsupported number of input bands %ld"),
                          path, shape[1]);
    }

    m->k_conv = ocl_kernel(ocl, "conv2d");
    m->k_dec = ocl_kernel(ocl, "decimate2");
    m->k_rx = ocl_kernel(ocl, "resize_x");
    m->k_ry = ocl_kernel(ocl, "resize_y");
    m->k_diff = ocl_kernel(ocl, "hc_diff");
    m->k_mask = ocl_kernel(ocl, "hc_mask");
    m->k_add = ocl_kernel(ocl, "hc_add");
    m->k_fft = ocl_kernel(ocl, "fft_lines");

    /* Tile buffers, sized for the largest step of the variant. */
    plane = (size_t)N * N * sizeof(float);
    max_side = m->variant == SR_RSWIR_X2 ? N : 4 * N;
    hr_plane = (size_t)max_side * max_side * sizeof(float);
    feat = m->rgbn.feat > m->f2.feat ? m->rgbn.feat : m->f2.feat;
    if (m->fus.feat > feat)
        feat = m->fus.feat;
    /* Only the fusion network of the main model runs at 2.5 m. */
    big = m->variant == SR_ALL_X4 ? hr_plane : plane;

    m->in = ocl_alloc(ocl, m->nin * plane, "input tile");
    m->ws_cat = ocl_alloc(ocl, 4 * feat * big, "features");
    m->ws_a = ocl_alloc(ocl, feat * big, "features");
    m->ws_b = ocl_alloc(ocl, feat * big, "features");
    m->rs_tmp = ocl_alloc(ocl, S2_NBANDS * (size_t)N * max_side * sizeof(float),
                          "resampling");
    m->lr_up = ocl_alloc(ocl, S2_NBANDS * hr_plane, "upsampled input");
    m->spec = ocl_alloc(ocl, S2_NBANDS * hr_plane * 2, "spectrum");
    if (m->variant != SR_RSWIR_X2) {
        m->x4 = ocl_alloc(ocl, 4 * plane, "RGBN tile");
        m->rgbn_sr = ocl_alloc(ocl, 4 * hr_plane, "RGBN output");
    }
    if (m->variant != SR_RGBN_X4) {
        m->x10 = ocl_alloc(ocl, 10 * plane, "10 m stack");
        m->sub = ocl_alloc(ocl, 6 * plane / 4, "20 m bands");
        m->f2out = ocl_alloc(ocl, 6 * plane, "20 m bands at 10 m");
    }
    if (m->variant == SR_ALL_X4) {
        m->fin = ocl_alloc(ocl, 10 * hr_plane, "fusion input");
        m->fus_out = ocl_alloc(ocl, 6 * hr_plane, "fusion output");
    }

    build_resize(ocl, &m->cubic4, N, 4 * N, 1);
    build_resize(ocl, &m->linear4, N, 4 * N, 0);
    build_resize(ocl, &m->linear2, N / 2, N, 0);
    build_resize(ocl, &m->cubic1, N, N, 1);
}

void sr_model_free(struct sr_model *m)
{
    cl_kernel *k[] = {&m->k_conv, &m->k_dec,  &m->k_rx,  &m->k_ry,
                      &m->k_diff, &m->k_mask, &m->k_add, &m->k_fft};
    cl_mem *b[] = {&m->in,         &m->x10,        &m->x4,
                   &m->sub,        &m->f2out,      &m->rgbn_sr,
                   &m->fin,        &m->fus_out,    &m->ws_cat,
                   &m->ws_a,       &m->ws_b,       &m->rs_tmp,
                   &m->lr_up,      &m->spec,       &m->hc_rgbn.mask,
                   &m->hc_f2.mask, &m->hc_fus.mask};
    size_t i;

    for (i = 0; i < sizeof(k) / sizeof(k[0]); i++)
        if (*k[i])
            clReleaseKernel(*k[i]);
    for (i = 0; i < sizeof(b) / sizeof(b[0]); i++)
        ocl_release(b[i]);
    free_cnn(&m->rgbn);
    free_cnn(&m->f2);
    free_cnn(&m->fus);
    free_resize(&m->cubic4);
    free_resize(&m->linear4);
    free_resize(&m->linear2);
    free_resize(&m->cubic1);
}

/* Inference. */

#define ARG(k, i, v) ocl_check(clSetKernelArg(k, i, sizeof(v), &v), #k)

static void launch(struct sr_model *m, cl_kernel k, int dims, size_t g0,
                   size_t g1, size_t g2)
{
    size_t global[3] = {g0, g1, g2};

    ocl_check(clEnqueueNDRangeKernel(m->ocl->queue, k, dims, NULL, global, NULL,
                                     0, NULL, NULL),
              "clEnqueueNDRangeKernel");
}

static void run_conv(struct sr_model *m, const struct sr_conv *c, cl_mem in,
                     int in_off, cl_mem out, int out_off, int n, int act,
                     int shuffle, cl_mem res, int res_off)
{
    cl_kernel k = m->k_conv;

    ARG(k, 0, in);
    ARG(k, 1, in_off);
    ARG(k, 2, c->cin);
    ARG(k, 3, c->w);
    ARG(k, 4, c->b);
    ARG(k, 5, out);
    ARG(k, 6, out_off);
    ARG(k, 7, c->cout);
    ARG(k, 8, n);
    ARG(k, 9, n);
    ARG(k, 10, c->k);
    ARG(k, 11, act);
    ARG(k, 12, shuffle);
    ARG(k, 13, res);
    ARG(k, 14, res_off);
    launch(m, k, 3, n, n, (c->cout + COB - 1) / COB);
}

/* CNNSR forward pass on an n x n tile, followed by the positivity clamp
 * that every SEN2SR wrapper applies to the network output. */
static void cnn_forward(struct sr_model *m, const struct sr_cnn *net, cl_mem in,
                        int in_off, cl_mem out, int n)
{
    int plane = n * n, F = net->feat;
    int s0 = 0, s1 = F * plane, s2 = 2 * F * plane, s3 = 3 * F * plane;
    cl_mem cat = m->ws_cat, a = m->ws_a, b = m->ws_b;

    /* Concatenation slots: 0 out_feature, 1 out_bn, 2 out_b1,
     * 3 out_blast. */
    run_conv(m, &net->conv1, in, in_off, cat, s0, n, ACT_NONE, 1, in, 0);

    /* First block: its output is out_b1. */
    run_conv(m, &net->first[0], cat, s0, a, 0, n, ACT_SILU, 1, a, 0);
    run_conv(m, &net->first[1], a, 0, b, 0, n, ACT_SILU, 1, b, 0);
    run_conv(m, &net->first[2], b, 0, cat, s2, n, ACT_ATTN, 1, cat, s0);

    /* Last block: silu(c1_r) is out_blast, its output feeds conv_2. */
    run_conv(m, &net->last[0], cat, s0, cat, s3, n, ACT_SILU, 1, cat, 0);
    run_conv(m, &net->last[1], cat, s3, b, 0, n, ACT_SILU, 1, b, 0);
    run_conv(m, &net->last[2], b, 0, a, 0, n, ACT_ATTN, 1, cat, s0);

    run_conv(m, &net->conv2, a, 0, cat, s1, n, ACT_NONE, 1, a, 0);
    run_conv(m, &net->cat, cat, 0, b, 0, n, ACT_NONE, 1, b, 0);
    run_conv(m, &net->ups, b, 0, out, 0, n, ACT_RELU, net->up, b, 0);
}

/* Resample the first c planes of src, (in x in), into dst, (out x out). */
static void resize(struct sr_model *m, const struct sr_resize *r, cl_mem src,
                   cl_mem dst, int c)
{
    cl_kernel k = m->k_rx;

    ARG(k, 0, src);
    ARG(k, 1, m->rs_tmp);
    ARG(k, 2, r->in);
    ARG(k, 3, r->in);
    ARG(k, 4, r->out);
    ARG(k, 5, r->idx);
    ARG(k, 6, r->wts);
    ARG(k, 7, r->kmax);
    launch(m, k, 3, r->out, r->in, c);

    k = m->k_ry;
    ARG(k, 0, m->rs_tmp);
    ARG(k, 1, dst);
    ARG(k, 2, r->in);
    ARG(k, 3, r->out);
    ARG(k, 4, r->out);
    ARG(k, 5, r->idx);
    ARG(k, 6, r->wts);
    ARG(k, 7, r->kmax);
    launch(m, k, 3, r->out, r->out, c);
}

static void fft2(struct sr_model *m, int c, int n, float sign)
{
    cl_kernel k = m->k_fft;
    size_t lsz = n / 2, wg = 0, global, local;
    int logn = 0, pass;

    while ((1 << logn) < n)
        logn++;
    clGetKernelWorkGroupInfo(k, m->ocl->device, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(wg), &wg, NULL);
    if (lsz > 256)
        lsz = 256;
    while (lsz > wg)
        lsz /= 2;
    global = (size_t)c * n * lsz;
    local = lsz;

    for (pass = 0; pass < 2; pass++) {
        int line_stride = pass == 0 ? n : 1, elem_stride = pass == 0 ? 1 : n;

        ARG(k, 0, m->spec);
        ARG(k, 1, n);
        ARG(k, 2, logn);
        ARG(k, 3, line_stride);
        ARG(k, 4, elem_stride);
        ARG(k, 5, sign);
        ocl_check(clEnqueueNDRangeKernel(m->ocl->queue, k, 1, NULL, &global,
                                         &local, 0, NULL, NULL),
                  "clEnqueueNDRangeKernel(fft_lines)");
    }
}

/* Hard constraint on the c planes of sr (n x n): take the low
 * frequencies from the first c planes of lr upsampled with the
 * anti-aliased bicubic filter r. */
static void hard_constraint(struct sr_model *m, const struct sr_hc *hc,
                            const struct sr_resize *r, cl_mem lr, cl_mem sr,
                            int c)
{
    int n = hc->n, plane = n * n, total = c * plane;
    float norm = 1.0f / plane;
    cl_kernel k;

    resize(m, r, lr, m->lr_up, c);

    k = m->k_diff;
    ARG(k, 0, m->lr_up);
    ARG(k, 1, sr);
    ARG(k, 2, m->spec);
    ARG(k, 3, total);
    launch(m, k, 1, total, 1, 1);

    fft2(m, c, n, -1.0f);

    k = m->k_mask;
    ARG(k, 0, m->spec);
    ARG(k, 1, hc->mask);
    ARG(k, 2, plane);
    ARG(k, 3, total);
    launch(m, k, 1, total, 1, 1);

    fft2(m, c, n, 1.0f);

    k = m->k_add;
    ARG(k, 0, sr);
    ARG(k, 1, m->spec);
    ARG(k, 2, norm);
    ARG(k, 3, total);
    launch(m, k, 1, total, 1, 1);
}

/* Copy plane src_c of src into plane dst_c of dst. */
static void copy_plane(struct sr_model *m, cl_mem src, int src_c, cl_mem dst,
                       int dst_c, size_t plane_bytes)
{
    ocl_copy(m->ocl, src, src_c * plane_bytes, dst, dst_c * plane_bytes,
             plane_bytes);
}

/* referencex2: the 20 m bands of the 10-band stack in (10 m grid,
 * S2 order) are restored at 10 m into f2out, (6, N, N). */
static void run_rswir_x2(struct sr_model *m, cl_mem in)
{
    static const int b20[6] = {B05, B06, B07, B8A, B11, B12};
    static const int b10[4] = {B02, B03, B04, B08};
    const int N = SR_TILE;
    size_t pb = (size_t)N * N * sizeof(float);
    cl_kernel k = m->k_dec;
    int i, h = N, w = N;

    /* Back to the 20 m grid by nearest-neighbour decimation, then
     * anti-aliased bilinear upsampling to 10 m. */
    for (i = 0; i < 6; i++)
        copy_plane(m, in, b20[i], m->f2out, i, pb);
    ARG(k, 0, m->f2out);
    ARG(k, 1, m->sub);
    ARG(k, 2, h);
    ARG(k, 3, w);
    launch(m, k, 3, N / 2, N / 2, 6);
    resize(m, &m->linear2, m->sub, m->x10, 6);
    for (i = 0; i < 4; i++)
        copy_plane(m, in, b10[i], m->x10, 6 + i, pb);

    cnn_forward(m, &m->f2, m->x10, 0, m->f2out, N);
    hard_constraint(m, &m->hc_f2, &m->cubic1, m->x10, m->f2out, 6);
}

/* nonreference: RGBN x4 of x4 (B04 B03 B02 B08) into rgbn_sr, same
 * band order. */
static void run_rgbn_x4(struct sr_model *m)
{
    cnn_forward(m, &m->rgbn, m->x4, 0, m->rgbn_sr, SR_TILE);
    hard_constraint(m, &m->hc_rgbn, &m->cubic4, m->x4, m->rgbn_sr, 4);
}

static void read_planes(struct sr_model *m, cl_mem src, const int *chan, int n,
                        float *dst, size_t plane)
{
    int i;

    for (i = 0; i < n; i++)
        ocl_read(m->ocl, src, chan[i] * plane * sizeof(float), dst + i * plane,
                 plane * sizeof(float));
}

void sr_model_run(struct sr_model *m, const float *in, float *out)
{
    const int N = SR_TILE, H = N * m->scale;
    size_t pb = (size_t)N * N * sizeof(float);
    size_t hp = (size_t)H * H, hpb = hp * sizeof(float);
    int i;

    ocl_write(m->ocl, m->in, 0, in, m->nin * pb);

    if (m->variant == SR_RGBN_X4) {
        /* Input B02 B03 B04 B08, network order B04 B03 B02 B08. */
        static const int to_net[4] = {2, 1, 0, 3};

        for (i = 0; i < 4; i++)
            copy_plane(m, m->in, to_net[i], m->x4, i, pb);
        run_rgbn_x4(m);
        read_planes(m, m->rgbn_sr, to_net, 4, out, hp);
    }
    else if (m->variant == SR_RSWIR_X2) {
        static const int all6[6] = {0, 1, 2, 3, 4, 5};

        run_rswir_x2(m, m->in);
        read_planes(m, m->f2out, all6, 6, out, hp);
    }
    else {
        /* referencex4: 20 m bands to 10 m, bilinear to 2.5 m, fused with
         * the super-resolved RGBN by the fusion network. */
        static const int to_net[4] = {B04, B03, B02, B08};
        static const int bgrn[4] = {2, 1, 0, 3};
        static const int from_fin[4] = {6, 7, 8, 9};
        static const int from_fus[6] = {0, 1, 2, 3, 4, 5};
        static const int out_rgbn[4] = {B02, B03, B04, B08};
        static const int out_rswir[6] = {B05, B06, B07, B8A, B11, B12};
        run_rswir_x2(m, m->in);
        resize(m, &m->linear4, m->f2out, m->fin, 6);

        for (i = 0; i < 4; i++)
            copy_plane(m, m->in, to_net[i], m->x4, i, pb);
        run_rgbn_x4(m);
        for (i = 0; i < 4; i++)
            copy_plane(m, m->rgbn_sr, bgrn[i], m->fin, 6 + i, hpb);

        cnn_forward(m, &m->fus, m->fin, 0, m->fus_out, H);
        hard_constraint(m, &m->hc_fus, &m->cubic4, m->f2out, m->fus_out, 6);

        for (i = 0; i < 4; i++)
            ocl_read(m->ocl, m->fin, from_fin[i] * hpb, out + out_rgbn[i] * hp,
                     hpb);
        for (i = 0; i < 6; i++)
            ocl_read(m->ocl, m->fus_out, from_fus[i] * hpb,
                     out + out_rswir[i] * hp, hpb);
    }
}
