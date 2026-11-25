#ifndef SEN2SR_OPENCL_NN_H
#define SEN2SR_OPENCL_NN_H

#include "ocl_backend.h"
#include "resize.h"
#include "safetensors.h"

/* Flat float32 tensor.  Dimensions are stored with singleton extents
 * padded on the left: [C] -> (1,C,1,1), [M,K] -> (1,M,K,1). */
struct tensor {
    int n, c, h, w;
    size_t count;
    cl_mem d;
};

struct nn {
    struct ocl_backend *ocl;
    int cpu;
    cl_mem dummy;
    cl_kernel conv, conv_cpu, convdw, gemm, norm, act, gather, lincomb,
        softmax, pool, scan, resize_x, resize_y, hc_diff, hc_mask, hc_add,
        fft, qkv_split, head_concat, mul, split2, proj_xdbl, proj_dts,
        combine, n2t, t2n, make_xs, split_xdbl, mul_last, gather_bands,
        concat2, clamp_pos, decimate, attn_qk, attn_bias, attn_av;
};

void nn_init(struct nn *nn, struct ocl_backend *ocl);
void nn_free(struct nn *nn);

struct tensor *tensor_alloc(struct ocl_backend *ocl, int n, int c, int h,
                            int w);
struct tensor *tensor_upload(struct ocl_backend *ocl, const float *host,
                             int n, int c, int h, int w);
/* Load a named F32 tensor and validate its shape (-1 matches any extent). */
struct tensor *tensor_load(struct ocl_backend *ocl, struct st_file *f,
                           const char *name, int ndim, long *shape);
void tensor_free(struct tensor *t);
void tensor_read(struct ocl_backend *ocl, struct tensor *t, float *host);
void tensor_zero(struct ocl_backend *ocl, struct tensor *t);

/* Activation codes (shared with the OpenCL kernels). */
enum { ACT_NONE = 0, ACT_SILU = 1, ACT_RELU = 3, ACT_GELU = 4,
       ACT_LEAKY = 5, ACT_SIGMOID = 6 };

/* Linear algebra / elementwise. */
void nn_gemm(struct nn *nn, struct tensor *a, struct tensor *b,
             struct tensor *bias, struct tensor *c, int M, int K, int N);
void nn_layernorm(struct nn *nn, struct tensor *in, struct tensor *out,
                  struct tensor *g, struct tensor *b, int M, int C, float eps);
void nn_act(struct nn *nn, struct tensor *in, struct tensor *out, int code,
            float p);
void nn_gather(struct nn *nn, struct tensor *in, struct tensor *out,
               cl_mem map, int count);
void nn_lincomb(struct nn *nn, struct tensor *a, struct tensor *b,
                struct tensor *out, float sa, float sb, float bias);
void nn_softmax(struct nn *nn, struct tensor *in, struct tensor *out,
                int rows, int N);
void nn_pool1(struct nn *nn, struct tensor *in, struct tensor *out, int C,
              int HW);
void nn_mul_bcast(struct nn *nn, struct tensor *x, struct tensor *s,
                  struct tensor *y, int C, int HW);
void nn_nchw2tokens(struct nn *nn, struct tensor *in, struct tensor *out,
                    int C, int L);
void nn_tokens2nchw(struct nn *nn, struct tensor *in, struct tensor *out,
                    int C, int L);
void nn_split2(struct nn *nn, struct tensor *x, struct tensor *a,
               struct tensor *b, int rows, int D);
void nn_scan(struct nn *nn, struct tensor *xs, struct tensor *dts,
             struct tensor *A, struct tensor *B, struct tensor *C,
             struct tensor *D, struct tensor *dt_bias, struct tensor *y,
             int Dinner, int N, int L);
void nn_proj_xdbl(struct nn *nn, struct tensor *xs, struct tensor *wt,
                  struct tensor *out, int K, int Dinner, int C, int L);
void nn_proj_dts(struct nn *nn, struct tensor *x, struct tensor *wt,
                 struct tensor *out, int K, int Dinner, int R, int L);
void nn_combine_y(struct nn *nn, struct tensor *out_y, struct tensor *y,
                  int Dinner, int H, int W);
void nn_make_xs(struct nn *nn, struct tensor *feat, struct tensor *xs,
                int Dinner, int H, int W);
void nn_split_xdbl(struct nn *nn, struct tensor *x, struct tensor *dts,
                   struct tensor *b, struct tensor *c, int K, int R, int N,
                   int L);
void nn_mul_last(struct nn *nn, struct tensor *x, struct tensor *s,
                 struct tensor *y, int C, int L);
void nn_gather_bands(struct nn *nn, struct tensor *in, struct tensor *out,
                     const int *idx, int K, int HW);
void nn_concat2(struct nn *nn, struct tensor *a, struct tensor *b,
                struct tensor *out, int CA, int K, int HW);
void nn_clamp(struct nn *nn, struct tensor *x);
void nn_decimate2(struct nn *nn, struct tensor *in, struct tensor *out,
                  int H, int W);

/* Resampling and Fourier hard constraint. */
void nn_resize(struct nn *nn, const struct resize_map *r, struct tensor *in,
               struct tensor *tmp, struct tensor *out, int C);
void nn_hc(struct nn *nn, struct tensor *lr_up, struct tensor *sr,
           cl_mem spec, cl_mem mask, int n, int C);

/* Attention helpers. */
void nn_qkv_split(struct nn *nn, struct tensor *qkv, struct tensor *q,
                  struct tensor *k, struct tensor *v, int nW, int heads,
                  int N, int C);
void nn_head_concat(struct nn *nn, struct tensor *in, struct tensor *out,
                    int nW, int heads, int N, int C);
void nn_attn_qk(struct nn *nn, struct tensor *q, struct tensor *k,
                struct tensor *scale, struct tensor *attn, int nW, int heads,
                int N, int d);
void nn_attn_bias(struct nn *nn, struct tensor *attn, struct tensor *rel,
                  struct tensor *mask, int nW, int heads, int N, int total);
void nn_attn_av(struct nn *nn, struct tensor *attn, struct tensor *v,
                struct tensor *out, int nW, int heads, int N, int d);

/* Convolution.  groups=1 uses the shared conv2d kernel; groups=cin is a
 * depthwise 3x3 convolution.  shuffle=1 stores normally; r>1 applies
 * pixel shuffle.  res may be NULL unless act == ACT_ATTN. */
void nn_conv(struct nn *nn, struct tensor *in, struct tensor *out,
             struct tensor *wt, struct tensor *bias, int groups, int act,
             int shuffle, struct tensor *res);

#endif /* SEN2SR_OPENCL_NN_H */
