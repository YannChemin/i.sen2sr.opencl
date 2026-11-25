#include "nn.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARG(k, i, v) do { \
    __typeof__(v) _v = (v); \
    ocl_check(clSetKernelArg((k), (i), sizeof(_v), &_v), "clSetKernelArg"); \
} while (0)
#define COB 4

/* Launch a logically flat kernel on a grid bounded by the device's per-axis
 * limits, in at most 1M-item chunks.  When base_arg >= 0 it is the kernel
 * argument through which the starting linear index is passed. */
static void launch_flat_ex(struct ocl_backend *ocl, cl_kernel k, size_t total,
                           const char *what, int base_arg)
{
    size_t maxdim[3], g[3] = {1, 1, 1}, local[3] = {1, 1, 1};
    const size_t chunk = 1000000;
    size_t base, left;
    int d;

    if (!total)
        return;
    ocl_check(clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                              sizeof(maxdim), maxdim, NULL),
              "CL_DEVICE_MAX_WORK_ITEM_SIZES");
    for (base = 0; base < total; base += chunk) {
        size_t n = total - base < chunk ? total - base : chunk;
        left = n;
        for (d = 0; d < 3; d++) {
            g[d] = left < maxdim[d] ? left : maxdim[d];
            left = (left + g[d] - 1) / g[d];
        }
        if (left != 1)
            die("kernel <%s> exceeds the device grid capacity", what);
        if (base_arg >= 0) {
            int b = (int)base;
            ocl_check(clSetKernelArg(k, (cl_uint)base_arg, sizeof(b), &b),
                      "clSetKernelArg(base)");
        }
        ocl_check(clEnqueueNDRangeKernel(ocl->queue, k, 3, NULL, g, local, 0,
                                         NULL, NULL), what);
    }
}

void nn_init(struct nn *nn, struct ocl_backend *ocl)
{
    static const float zero[4] = {0, 0, 0, 0};

    memset(nn, 0, sizeof(*nn));
    nn->ocl = ocl;
    nn->cpu = ocl->arch == OCL_ARCH_CPU;
    nn->dummy = ocl_upload(ocl, zero, sizeof(zero), "zero buffer");
    nn->conv = ocl_kernel(ocl, "conv2d_full");
    nn->conv_cpu = ocl_kernel(ocl, "conv2d_cpu_full");
    nn->convdw = ocl_kernel(ocl, "conv_dw3");
    nn->gemm = ocl_kernel(ocl, "gemm");
    nn->norm = ocl_kernel(ocl, "layernorm");
    nn->act = ocl_kernel(ocl, "act_k");
    nn->gather = ocl_kernel(ocl, "gather_k");
    nn->lincomb = ocl_kernel(ocl, "lincomb");
    nn->softmax = ocl_kernel(ocl, "softmax_k");
    nn->pool = ocl_kernel(ocl, "pool1");
    nn->scan = ocl_kernel(ocl, "selective_scan");
    nn->resize_x = ocl_kernel(ocl, "resize_x_full");
    nn->resize_y = ocl_kernel(ocl, "resize_y_full");
    nn->hc_diff = ocl_kernel(ocl, "hc_diff_full");
    nn->hc_mask = ocl_kernel(ocl, "hc_mask_full");
    nn->hc_add = ocl_kernel(ocl, "hc_add_full");
    nn->fft = ocl_kernel(ocl, "fft_lines_full");
    nn->qkv_split = ocl_kernel(ocl, "qkv_split");
    nn->head_concat = ocl_kernel(ocl, "head_concat");
    nn->mul = ocl_kernel(ocl, "mul_bcast");
    nn->split2 = ocl_kernel(ocl, "split2");
    nn->proj_xdbl = ocl_kernel(ocl, "proj_xdbl");
    nn->proj_dts = ocl_kernel(ocl, "proj_dts");
    nn->combine = ocl_kernel(ocl, "combine_y");
    nn->n2t = ocl_kernel(ocl, "nchw2tokens");
    nn->t2n = ocl_kernel(ocl, "tokens2nchw");
    nn->make_xs = ocl_kernel(ocl, "make_xs");
    nn->split_xdbl = ocl_kernel(ocl, "split_xdbl");
    nn->mul_last = ocl_kernel(ocl, "mul_bcast_last");
    nn->gather_bands = ocl_kernel(ocl, "gather_bands");
    nn->concat2 = ocl_kernel(ocl, "concat2");
    nn->clamp_pos = ocl_kernel(ocl, "clamp_pos");
    nn->decimate = ocl_kernel(ocl, "decimate2_full");
    nn->attn_qk = ocl_kernel(ocl, "attn_qk");
    nn->attn_bias = ocl_kernel(ocl, "attn_add_bias");
    nn->attn_av = ocl_kernel(ocl, "attn_av");
}

void nn_free(struct nn *nn)
{
    ocl_release(&nn->dummy);
}

static struct tensor *tensor_new(struct ocl_backend *ocl, int n, int c, int h,
                                 int w)
{
    struct tensor *t = calloc(1, sizeof(*t));
    if (!t)
        die("out of memory");
    t->n = n;
    t->c = c;
    t->h = h;
    t->w = w;
    t->count = (size_t)n * c * h * w;
    t->d = ocl_alloc(ocl, t->count * sizeof(float), "tensor");
    return t;
}

struct tensor *tensor_alloc(struct ocl_backend *ocl, int n, int c, int h,
                            int w)
{
    return tensor_new(ocl, n, c, h, w);
}

struct tensor *tensor_upload(struct ocl_backend *ocl, const float *host,
                             int n, int c, int h, int w)
{
    struct tensor *t = tensor_new(ocl, n, c, h, w);
    if (t->count)
        ocl_write(ocl, t->d, 0, host, t->count * sizeof(float));
    return t;
}

struct tensor *tensor_load(struct ocl_backend *ocl, struct st_file *f,
                           const char *name, int ndim, long *shape)
{
    long got[ST_MAX_DIMS];
    float *host;
    struct tensor *t;
    int i;

    host = st_get_f32(f, name, ndim, shape);
    for (i = 0; i < ST_MAX_DIMS; i++)
        got[i] = 1;
    for (i = 0; i < ndim && i < 4; i++)
        got[i] = shape[i];
    t = tensor_upload(ocl, host, got[0], got[1], got[2], got[3]);
    free(host);
    return t;
}

void tensor_free(struct tensor *t)
{
    if (!t)
        return;
    ocl_release(&t->d);
    free(t);
}

void tensor_read(struct ocl_backend *ocl, struct tensor *t, float *host)
{
    if (t->count)
        ocl_read(ocl, t->d, 0, host, t->count * sizeof(float));
}

void tensor_zero(struct ocl_backend *ocl, struct tensor *t)
{
    float *z;
    if (!t->count)
        return;
    z = calloc(t->count, sizeof(float));
    if (!z)
        die("out of memory");
    ocl_write(ocl, t->d, 0, z, t->count * sizeof(float));
    free(z);
}

void nn_gemm(struct nn *nn, struct tensor *a, struct tensor *b,
             struct tensor *bias, struct tensor *c, int M, int K, int N)
{
    cl_kernel k = nn->gemm;
    cl_int has = bias ? 1 : 0;
    cl_mem bptr = bias ? bias->d : nn->dummy;

    ARG(k, 0, a->d);
    ARG(k, 1, b->d);
    ARG(k, 2, bptr);
    ARG(k, 3, has);
    ARG(k, 4, c->d);
    ARG(k, 5, M);
    ARG(k, 6, K);
    ARG(k, 7, N);
    ARG(k, 8, 0); /* row_base: base is applied to the flat index */
    launch_flat_ex(nn->ocl, k, (size_t)M * (size_t)N, "gemm", 9);
}

void nn_layernorm(struct nn *nn, struct tensor *in, struct tensor *out,
                  struct tensor *g, struct tensor *b, int M, int C, float eps)
{
    cl_kernel k = nn->norm;
    ARG(k, 0, in->d);
    ARG(k, 1, out->d);
    ARG(k, 2, g->d);
    ARG(k, 3, b->d);
    ARG(k, 4, M);
    ARG(k, 5, C);
    ARG(k, 6, eps);
    ARG(k, 7, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)M, "layernorm", 7);
}

void nn_act(struct nn *nn, struct tensor *in, struct tensor *out, int code,
            float p)
{
    cl_kernel k = nn->act;
    ARG(k, 0, in->d);
    ARG(k, 1, out->d);
    ARG(k, 2, (int)in->count);
    ARG(k, 3, code);
    ARG(k, 4, p);
    ARG(k, 5, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, in->count, "act", 5);
}

void nn_gather(struct nn *nn, struct tensor *in, struct tensor *out,
               cl_mem map, int count)
{
    cl_kernel k = nn->gather;
    ARG(k, 0, in->d);
    ARG(k, 1, out->d);
    ARG(k, 2, map);
    ARG(k, 3, count);
    ARG(k, 4, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)count, "gather", 4);
}

void nn_lincomb(struct nn *nn, struct tensor *a, struct tensor *b,
                struct tensor *out, float sa, float sb, float bias)
{
    cl_kernel k = nn->lincomb;
    ARG(k, 0, a->d);
    ARG(k, 1, b ? b->d : nn->dummy);
    ARG(k, 2, out->d);
    ARG(k, 3, (int)out->count);
    ARG(k, 4, sa);
    ARG(k, 5, sb);
    ARG(k, 6, bias);
    ARG(k, 7, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, out->count, "lincomb", 7);
}

void nn_softmax(struct nn *nn, struct tensor *in, struct tensor *out,
                int rows, int N)
{
    cl_kernel k = nn->softmax;
    ARG(k, 0, in->d);
    ARG(k, 1, out->d);
    ARG(k, 2, rows);
    ARG(k, 3, N);
    ARG(k, 4, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)rows, "softmax", 4);
}

void nn_pool1(struct nn *nn, struct tensor *in, struct tensor *out, int C,
              int HW)
{
    cl_kernel k = nn->pool;
    ARG(k, 0, in->d);
    ARG(k, 1, out->d);
    ARG(k, 2, C);
    ARG(k, 3, HW);
    ARG(k, 4, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)C, "pool1", 4);
}

void nn_mul_bcast(struct nn *nn, struct tensor *x, struct tensor *s,
                  struct tensor *y, int C, int HW)
{
    cl_kernel k = nn->mul;
    ARG(k, 0, x->d);
    ARG(k, 1, s->d);
    ARG(k, 2, y->d);
    ARG(k, 3, C);
    ARG(k, 4, HW);
    ARG(k, 5, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)HW * (size_t)C, "mul_bcast", 5);
}

void nn_nchw2tokens(struct nn *nn, struct tensor *in, struct tensor *out,
                    int C, int L)
{
    cl_kernel k = nn->n2t;
    ARG(k, 0, in->d);
    ARG(k, 1, out->d);
    ARG(k, 2, C);
    ARG(k, 3, L);
    ARG(k, 4, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)L * (size_t)C, "nchw2tokens", 4);
}

void nn_tokens2nchw(struct nn *nn, struct tensor *in, struct tensor *out,
                    int C, int L)
{
    cl_kernel k = nn->t2n;
    ARG(k, 0, in->d);
    ARG(k, 1, out->d);
    ARG(k, 2, C);
    ARG(k, 3, L);
    ARG(k, 4, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)L * (size_t)C, "tokens2nchw", 4);
}

void nn_split2(struct nn *nn, struct tensor *x, struct tensor *a,
               struct tensor *b, int rows, int D)
{
    cl_kernel k = nn->split2;
    ARG(k, 0, x->d);
    ARG(k, 1, a->d);
    ARG(k, 2, b->d);
    ARG(k, 3, rows);
    ARG(k, 4, D);
    ARG(k, 5, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)rows * (size_t)D, "split2", 5);
}

void nn_scan(struct nn *nn, struct tensor *xs, struct tensor *dts,
             struct tensor *A, struct tensor *B, struct tensor *C,
             struct tensor *D, struct tensor *dt_bias, struct tensor *y,
             int Dinner, int N, int L)
{
    cl_kernel k = nn->scan;
    ARG(k, 0, xs->d);
    ARG(k, 1, dts->d);
    ARG(k, 2, A->d);
    ARG(k, 3, B->d);
    ARG(k, 4, C->d);
    ARG(k, 5, D->d);
    ARG(k, 6, dt_bias->d);
    ARG(k, 7, y->d);
    ARG(k, 8, Dinner);
    ARG(k, 9, N);
    ARG(k, 10, L);
    ARG(k, 11, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)Dinner * 4, "selective_scan", 11);
}

void nn_proj_xdbl(struct nn *nn, struct tensor *xs, struct tensor *wt,
                  struct tensor *out, int K, int Dinner, int C, int L)
{
    cl_kernel k = nn->proj_xdbl;
    ARG(k, 0, xs->d);
    ARG(k, 1, wt->d);
    ARG(k, 2, out->d);
    ARG(k, 3, K);
    ARG(k, 4, Dinner);
    ARG(k, 5, C);
    ARG(k, 6, L);
    ARG(k, 7, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)K * (size_t)C * (size_t)L, "proj_xdbl", 7);
}

void nn_proj_dts(struct nn *nn, struct tensor *x, struct tensor *wt,
                 struct tensor *out, int K, int Dinner, int R, int L)
{
    cl_kernel k = nn->proj_dts;
    ARG(k, 0, x->d);
    ARG(k, 1, wt->d);
    ARG(k, 2, out->d);
    ARG(k, 3, K);
    ARG(k, 4, Dinner);
    ARG(k, 5, R);
    ARG(k, 6, L);
    ARG(k, 7, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)K * (size_t)Dinner * (size_t)L, "proj_dts", 7);
}

void nn_combine_y(struct nn *nn, struct tensor *out_y, struct tensor *y,
                  int Dinner, int H, int W)
{
    cl_kernel k = nn->combine;
    ARG(k, 0, out_y->d);
    ARG(k, 1, y->d);
    ARG(k, 2, Dinner);
    ARG(k, 3, H);
    ARG(k, 4, W);
    ARG(k, 5, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)Dinner * H * W, "combine_y", 5);
}

void nn_make_xs(struct nn *nn, struct tensor *feat, struct tensor *xs,
                int Dinner, int H, int W)
{
    cl_kernel k = nn->make_xs;
    ARG(k, 0, feat->d);
    ARG(k, 1, xs->d);
    ARG(k, 2, Dinner);
    ARG(k, 3, H);
    ARG(k, 4, W);
    ARG(k, 5, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, 4 * (size_t)Dinner * (size_t)H * (size_t)W, "make_xs", 5);
}

void nn_split_xdbl(struct nn *nn, struct tensor *x, struct tensor *dts,
                   struct tensor *b, struct tensor *c, int K, int R, int N,
                   int L)
{
    cl_kernel k = nn->split_xdbl;
    ARG(k, 0, x->d);
    ARG(k, 1, dts->d);
    ARG(k, 2, b->d);
    ARG(k, 3, c->d);
    ARG(k, 4, K);
    ARG(k, 5, R);
    ARG(k, 6, N);
    ARG(k, 7, L);
    ARG(k, 8, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)K * (size_t)(R + 2 * N) * (size_t)L, "split_xdbl", 8);
}

void nn_mul_last(struct nn *nn, struct tensor *x, struct tensor *s,
                 struct tensor *y, int C, int L)
{
    cl_kernel k = nn->mul_last;
    ARG(k, 0, x->d);
    ARG(k, 1, s->d);
    ARG(k, 2, y->d);
    ARG(k, 3, C);
    ARG(k, 4, L);
    ARG(k, 5, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)L * (size_t)C, "mul_bcast_last", 5);
}

void nn_gather_bands(struct nn *nn, struct tensor *in, struct tensor *out,
                     const int *idx, int K, int HW)
{
    cl_kernel k = nn->gather_bands;
    cl_mem map = ocl_upload(nn->ocl, idx, (size_t)K * sizeof(int), "band map");
    ARG(k, 0, in->d);
    ARG(k, 1, out->d);
    ARG(k, 2, map);
    ARG(k, 3, HW);
    ARG(k, 4, K);
    ARG(k, 5, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)HW * (size_t)K, "gather_bands", 5);
    clFinish(nn->ocl->queue);
    ocl_release(&map);
}

void nn_concat2(struct nn *nn, struct tensor *a, struct tensor *b,
                struct tensor *out, int CA, int K, int HW)
{
    cl_kernel k = nn->concat2;
    ARG(k, 0, a->d);
    ARG(k, 1, b->d);
    ARG(k, 2, out->d);
    ARG(k, 3, CA);
    ARG(k, 4, K);
    ARG(k, 5, HW);
    ARG(k, 6, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)HW * (size_t)(CA + K), "concat2", 6);
}

void nn_clamp(struct nn *nn, struct tensor *x)
{
    cl_kernel k = nn->clamp_pos;
    ARG(k, 0, x->d);
    ARG(k, 1, (int)x->count);
    ARG(k, 2, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, x->count, "clamp_pos", 2);
}

void nn_decimate2(struct nn *nn, struct tensor *in, struct tensor *out,
                  int H, int W)
{
    cl_kernel k = nn->decimate;
    ARG(k, 0, in->d);
    ARG(k, 1, out->d);
    ARG(k, 2, H);
    ARG(k, 3, W);
    ARG(k, 4, in->c);
    ARG(k, 5, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)(W / 2) * (H / 2) * in->c,
                   "decimate2_full", 5);
}

void nn_resize(struct nn *nn, const struct resize_map *r, struct tensor *in,
               struct tensor *tmp, struct tensor *out, int C)
{
    cl_kernel k = nn->resize_x;
    ARG(k, 0, in->d);
    ARG(k, 1, tmp->d);
    ARG(k, 2, r->in);
    ARG(k, 3, r->in);
    ARG(k, 4, r->out);
    ARG(k, 5, C);
    ARG(k, 6, r->idx_x);
    ARG(k, 7, r->wts_x);
    ARG(k, 8, r->kmax);
    ARG(k, 9, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)r->out * r->in * C, "resize_x_full", 9);

    k = nn->resize_y;
    ARG(k, 0, tmp->d);
    ARG(k, 1, out->d);
    ARG(k, 2, r->in);
    ARG(k, 3, r->out);
    ARG(k, 4, r->out);
    ARG(k, 5, C);
    ARG(k, 6, r->idx_y);
    ARG(k, 7, r->wts_y);
    ARG(k, 8, r->kmax);
    ARG(k, 9, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)r->out * r->out * C, "resize_y_full", 9);
}

static void fft2(struct nn *nn, cl_mem spec, int C, int n, float sign)
{
    size_t lsz = n / 2, wg = 0, global[3], local[3], maxdim[3];
    size_t lines, blocks;
    int logn = 0, pass;

    while ((1 << logn) < n)
        logn++;
    clGetKernelWorkGroupInfo(nn->fft, nn->ocl->device, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(wg), &wg, NULL);
    if (lsz > 256)
        lsz = 256;
    while (lsz > wg)
        lsz /= 2;
    ocl_check(clGetDeviceInfo(nn->ocl->device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                              sizeof(maxdim), maxdim, NULL),
              "CL_DEVICE_MAX_WORK_ITEM_SIZES");
    lines = (size_t)C * (size_t)n;
    global[0] = lsz;
    global[1] = lines < maxdim[1] ? lines : maxdim[1];
    blocks = (lines + global[1] - 1) / global[1];
    if (blocks > maxdim[2])
        die("fft grid too large");
    global[2] = blocks;
    local[0] = lsz;
    local[1] = 1;
    local[2] = 1;

    for (pass = 0; pass < 2; pass++) {
        int line_stride = pass == 0 ? n : 1;
        int elem_stride = pass == 0 ? 1 : n;
        cl_kernel k = nn->fft;

        ARG(k, 0, spec);
        ARG(k, 1, n);
        ARG(k, 2, logn);
        ARG(k, 3, line_stride);
        ARG(k, 4, elem_stride);
        ARG(k, 5, sign);
        ocl_check(clEnqueueNDRangeKernel(nn->ocl->queue, k, 3, NULL, global,
                                         local, 0, NULL, NULL), "fft_lines_full");
    }
}

void nn_hc(struct nn *nn, struct tensor *lr_up, struct tensor *sr,
           cl_mem spec, cl_mem mask, int n, int C)
{
    int plane = n * n, total = C * plane;
    float norm = 1.0f / plane;
    cl_kernel k;

    k = nn->hc_diff;
    ARG(k, 0, lr_up->d);
    ARG(k, 1, sr->d);
    ARG(k, 2, spec);
    ARG(k, 3, total);
    ARG(k, 4, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)total, "hc_diff_full", 4);
    fft2(nn, spec, C, n, -1.0f);

    k = nn->hc_mask;
    ARG(k, 0, spec);
    ARG(k, 1, mask);
    ARG(k, 2, plane);
    ARG(k, 3, total);
    ARG(k, 4, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)total, "hc_mask_full", 4);
    fft2(nn, spec, C, n, 1.0f);

    k = nn->hc_add;
    ARG(k, 0, sr->d);
    ARG(k, 1, spec);
    ARG(k, 2, norm);
    ARG(k, 3, total);
    ARG(k, 4, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, k, (size_t)total, "hc_add_full", 4);
}

void nn_qkv_split(struct nn *nn, struct tensor *qkv, struct tensor *q,
                  struct tensor *k, struct tensor *v, int nW, int heads,
                  int N, int C)
{
    cl_kernel kk = nn->qkv_split;
    ARG(kk, 0, qkv->d);
    ARG(kk, 1, q->d);
    ARG(kk, 2, k->d);
    ARG(kk, 3, v->d);
    ARG(kk, 4, nW);
    ARG(kk, 5, heads);
    ARG(kk, 6, N);
    ARG(kk, 7, C);
    ARG(kk, 8, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, kk, (size_t)nW * heads * N, "qkv_split", 8);
}

void nn_head_concat(struct nn *nn, struct tensor *in, struct tensor *out,
                    int nW, int heads, int N, int C)
{
    cl_kernel kk = nn->head_concat;
    ARG(kk, 0, in->d);
    ARG(kk, 1, out->d);
    ARG(kk, 2, nW);
    ARG(kk, 3, heads);
    ARG(kk, 4, N);
    ARG(kk, 5, C);
    ARG(kk, 6, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, kk, (size_t)nW * heads * N, "head_concat", 6);
}

void nn_attn_qk(struct nn *nn, struct tensor *q, struct tensor *k,
                struct tensor *scale, struct tensor *attn, int nW, int heads,
                int N, int d)
{
    cl_kernel kk = nn->attn_qk;
    ARG(kk, 0, q->d);
    ARG(kk, 1, k->d);
    ARG(kk, 2, scale->d);
    ARG(kk, 3, attn->d);
    ARG(kk, 4, nW);
    ARG(kk, 5, heads);
    ARG(kk, 6, N);
    ARG(kk, 7, d);
    ARG(kk, 8, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, kk, (size_t)nW * heads * N, "attn_qk", 8);
}

void nn_attn_bias(struct nn *nn, struct tensor *attn, struct tensor *rel,
                  struct tensor *mask, int nW, int heads, int N, int total)
{
    cl_kernel kk = nn->attn_bias;
    ARG(kk, 0, attn->d);
    ARG(kk, 1, rel->d);
    ARG(kk, 2, mask->d);
    ARG(kk, 3, nW);
    ARG(kk, 4, heads);
    ARG(kk, 5, N);
    ARG(kk, 6, total);
    {
        size_t g[3] = {(size_t)N, (size_t)N, (size_t)heads};
        ocl_check(clEnqueueNDRangeKernel(nn->ocl->queue, kk, 3, NULL, g, NULL,
                                         0, NULL, NULL), "attn_bias");
    }
}

void nn_attn_av(struct nn *nn, struct tensor *attn, struct tensor *v,
                struct tensor *out, int nW, int heads, int N, int d)
{
    cl_kernel kk = nn->attn_av;
    ARG(kk, 0, attn->d);
    ARG(kk, 1, v->d);
    ARG(kk, 2, out->d);
    ARG(kk, 3, nW);
    ARG(kk, 4, heads);
    ARG(kk, 5, N);
    ARG(kk, 6, d);
    ARG(kk, 7, 0); /* set per chunk */
    launch_flat_ex(nn->ocl, kk, (size_t)nW * heads * N, "attn_av", 7);
}

void nn_conv(struct nn *nn, struct tensor *in, struct tensor *out,
             struct tensor *wt, struct tensor *bias, int groups, int act,
             int shuffle, struct tensor *res)
{
    struct ocl_backend *ocl = nn->ocl;
    cl_kernel k;
    int cin = in->c, cout = wt->n;
    int ksz = wt->h, H = in->h, W = in->w;
    cl_mem resp = res ? res->d : nn->dummy;

    if (getenv("SEN2SR_DEBUG_CONV"))
        fprintf(stderr, "conv cin=%d cout=%d H=%d W=%d k=%d act=%d shuffle=%d\n",
                cin, cout, H, W, ksz, act, shuffle);

    if (groups == cin) {
        k = nn->convdw;
        ARG(k, 0, in->d);
        ARG(k, 1, wt->d);
        ARG(k, 2, bias ? bias->d : nn->dummy);
        ARG(k, 3, out->d);
        ARG(k, 4, cin);
        ARG(k, 5, H);
        ARG(k, 6, W);
        ARG(k, 7, 0); /* set per chunk */
        launch_flat_ex(ocl, k, (size_t)cin * H * W, "conv_dw3", 7);
        if (act != ACT_NONE)
            nn_act(nn, out, out, act, 0.01f);
        return;
    }

    k = nn->cpu ? nn->conv_cpu : nn->conv;
    ARG(k, 0, in->d);
    ARG(k, 1, 0);
    ARG(k, 2, 0);
    ARG(k, 3, cin);
    ARG(k, 4, wt->d);
    ARG(k, 5, bias ? bias->d : nn->dummy);
    ARG(k, 6, out->d);
    ARG(k, 7, 0);
    ARG(k, 8, 0);
    ARG(k, 9, cout);
    ARG(k, 10, H);
    ARG(k, 11, W);
    ARG(k, 12, ksz);
    ARG(k, 13, act);
    ARG(k, 14, shuffle);
    ARG(k, 15, resp);
    ARG(k, 16, 0);
    ARG(k, 17, 0);
    ARG(k, 18, 0); /* set per chunk */
    {
        size_t blocks = (size_t)((cout + COB - 1) / COB);
        size_t total = nn->cpu ? (size_t)((W + 7) / 8) * H * blocks
                               : (size_t)W * H * blocks;
        launch_flat_ex(ocl, k, total, "conv2d_full", 18);
    }
    if (getenv("SEN2SR_DEBUG_CONV"))
        clFinish(ocl->queue);
}
