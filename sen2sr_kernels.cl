/* OpenCL C 1.1 kernels for i.sen2sr.opencl.
 *
 * Tensors are float32, channel-major (C, H, W) planes, one tile at a time.
 * Buffer arguments come with an element offset so that a kernel can read
 * or write a channel slice of a larger buffer (e.g. the concatenation
 * buffer of the CNN). */

#define ACT_NONE 0
#define ACT_SILU 1
#define ACT_ATTN 2
#define ACT_RELU 3
#define ACT_GELU 4
#define ACT_LEAKY 5
#define ACT_SIGMOID 6

/* Number of output channels computed per work-item by conv2d. */
#define COB 4

/* Bounded flat launch helper used by the kernels of the full SEN2SR
 * pipeline, which need grids larger than a device's per-axis limit. */
#define FLAT_RAW (get_global_id(0) + get_global_size(0) * \
                  (get_global_id(1) + get_global_size(1) * get_global_id(2)))
#define FLAT_ID (FLAT_RAW + base)

/* Feature maps may be stored as IEEE half (precision=fp16): conv2d takes
 * its input, output and residual as untyped pointers with a flag each,
 * and converts with the core vload_half/vstore_half (no cl_khr_fp16
 * needed). Arithmetic is float either way. */
float ld(__global const void *p, int i, int half_)
{
    return half_ ? vload_half(i, (__global const half *)p)
                 : ((__global const float *)p)[i];
}

void st(__global void *p, int i, int half_, float v)
{
    if (half_)
        vstore_half(v, i, (__global half *)p);
    else
        ((__global float *)p)[i] = v;
}

/* Epilogue of conv2d and conv2d_cpu: activation, then the store, through
 * the pixel shuffle when r > 1. */
void conv_store(__global void *out, int out_off, int out_h,
                __global const void *res, int res_off, int res_h, int co,
                int x, int y, int h, int w, int act, int shuffle, float v)
{
    int plane = h * w;

    if (act == ACT_SILU)
        v = v / (1.0f + exp(-v));
    else if (act == ACT_ATTN)
        v = (v + ld(res, res_off + co * plane + y * w + x, res_h)) *
            (1.0f / (1.0f + exp(-v)) - 0.5f);
    else if (act == ACT_RELU)
        v = fmax(v, 0.0f);
    else if (act == ACT_GELU)
        v = 0.5f * v * (1.0f + erf(v * 0.7071067811865476f));
    else if (act == ACT_LEAKY)
        v = v >= 0.0f ? v : 0.01f * v;
    else if (act == ACT_SIGMOID)
        v = 1.0f / (1.0f + exp(-v));

    if (shuffle == 1)
        st(out, out_off + co * plane + y * w + x, out_h, v);
    else {
        int r2 = shuffle * shuffle, c = co / r2, sub = co % r2;
        int oy = y * shuffle + sub / shuffle;
        int ox = x * shuffle + sub % shuffle;

        st(out, out_off + (c * h * shuffle + oy) * w * shuffle + ox, out_h, v);
    }
}

/* k x k convolution (k = 1 or 3), stride 1, zero padding k / 2, with a
 * fused epilogue:
 *   ACT_SILU  y = x * sigmoid(x)
 *   ACT_ATTN  y = (x + res) * (sigmoid(x) - 0.5), the SPAB attention
 *   ACT_RELU  y = max(x, 0), the SEN2SR positivity clamp
 * With shuffle r > 1 the result is written through a pixel shuffle:
 * channel co goes to plane co / r^2 at sub-pixel ((co % r^2) / r,
 * co % r) of an (h * r, w * r) output.
 * Weights are (cout, cin, k, k), bias (cout). in_h, out_h and res_h say
 * whether in, out and res hold half values. Global size
 * (w, h, ceil(cout / COB)). */
__kernel void conv2d(__global const void *in, int in_off, int in_h, int cin,
                     __global const float *wt, __global const float *bias,
                     __global void *out, int out_off, int out_h, int cout,
                     int h, int w, int k, int act, int shuffle,
                     __global const void *res, int res_off, int res_h)
{
    int x = get_global_id(0), y = get_global_id(1);
    int co0 = get_global_id(2) * COB;
    int plane = h * w, pad = k / 2, kk = k * k;
    float acc[COB];
    int j, ci, dy, dx;

    if (x >= w || y >= h)
        return;
    for (j = 0; j < COB; j++)
        acc[j] = co0 + j < cout ? bias[co0 + j] : 0.0f;

    for (ci = 0; ci < cin; ci++) {
        int src = in_off + ci * plane;

        for (dy = 0; dy < k; dy++) {
            int yy = y + dy - pad;

            if (yy < 0 || yy >= h)
                continue;
            for (dx = 0; dx < k; dx++) {
                int xx = x + dx - pad;
                float v;

                if (xx < 0 || xx >= w)
                    continue;
                v = ld(in, src + yy * w + xx, in_h);
                for (j = 0; j < COB; j++)
                    if (co0 + j < cout)
                        acc[j] +=
                            v * wt[((co0 + j) * cin + ci) * kk + dy * k + dx];
            }
        }
    }

    for (j = 0; j < COB && co0 + j < cout; j++)
        conv_store(out, out_off, out_h, res, res_off, res_h, co0 + j, x, y, h,
                   w, act, shuffle, acc[j]);
}

/* Row segment of 8 pixels starting at column x0 of row yy, zero outside
 * the plane (columns < 0 or >= w). */
float8 ld_row8(__global const void *in, int base, int x0, int w, int in_h)
{
    float v[8];
    int i;

    if (x0 >= 0 && x0 + 8 <= w)
        return in_h ? vload_half8(0, (__global const half *)in + base + x0)
                    : vload8(0, (__global const float *)in + base + x0);
    for (i = 0; i < 8; i++)
        v[i] = x0 + i >= 0 && x0 + i < w ? ld(in, base + x0 + i, in_h) : 0.0f;
    return (float8)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

/* conv2d for CPU devices (profile=cpu): same contract, but each
 * work-item computes 8 consecutive pixels of COB output channels with
 * 8-wide vector loads along the row, which PoCL maps onto the SIMD
 * units. Global size (ceil(w / 8), h, ceil(cout / COB)). */
__kernel void conv2d_cpu(__global const void *in, int in_off, int in_h,
                         int cin, __global const float *wt,
                         __global const float *bias, __global void *out,
                         int out_off, int out_h, int cout, int h, int w, int k,
                         int act, int shuffle, __global const void *res,
                         int res_off, int res_h)
{
    int x0 = get_global_id(0) * 8, y = get_global_id(1);
    int co0 = get_global_id(2) * COB;
    int plane = h * w, pad = k / 2, kk = k * k;
    float8 acc[COB];
    int j, ci, dy, dx, i;

    if (x0 >= w || y >= h)
        return;
    for (j = 0; j < COB; j++)
        acc[j] = (float8)(co0 + j < cout ? bias[co0 + j] : 0.0f);

    for (ci = 0; ci < cin; ci++) {
        int src = in_off + ci * plane;

        for (dy = 0; dy < k; dy++) {
            int yy = y + dy - pad;

            if (yy < 0 || yy >= h)
                continue;
            for (dx = 0; dx < k; dx++) {
                const float8 v = ld_row8(in, src + yy * w, x0 + dx - pad, w,
                                         in_h);

                for (j = 0; j < COB; j++)
                    if (co0 + j < cout)
                        acc[j] = mad(
                            (float8)(wt[((co0 + j) * cin + ci) * kk + dy * k +
                                        dx]),
                            v, acc[j]);
            }
        }
    }

    for (j = 0; j < COB && co0 + j < cout; j++) {
        const float a[8] = {acc[j].s0, acc[j].s1, acc[j].s2, acc[j].s3,
                            acc[j].s4, acc[j].s5, acc[j].s6, acc[j].s7};

        for (i = 0; i < 8 && x0 + i < w; i++)
            conv_store(out, out_off, out_h, res, res_off, res_h, co0 + j,
                       x0 + i, y, h, w, act, shuffle, a[i]);
    }
}

/* Nearest-neighbour 2x decimation keeping even rows and columns, as
 * torch interpolate(scale_factor=0.5, mode="nearest"). Global size
 * (w / 2, h / 2, c). */
__kernel void decimate2(__global const float *in, __global float *out, int h,
                        int w)
{
    int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    int ho = h / 2, wo = w / 2;

    if (x >= wo || y >= ho)
        return;
    out[(c * ho + y) * wo + x] = in[(c * h + 2 * y) * w + 2 * x];
}

/* Separable anti-aliased resampling, one axis at a time, with the
 * per-output-index support start/length (idx) and normalised weights
 * (wts, kmax per index) precomputed on the host as in PyTorch's
 * antialias=True path. */

/* Along x: (c, h, win) -> (c, h, wout). Global size (wout, h, c). */
__kernel void resize_x(__global const float *in, __global float *out, int h,
                       int win, int wout, __global const int *idx,
                       __global const float *wts, int kmax)
{
    int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    __global const float *row;
    float acc = 0.0f;
    int j, x0, n;

    if (x >= wout || y >= h)
        return;
    x0 = idx[2 * x];
    n = idx[2 * x + 1];
    row = in + (c * h + y) * win + x0;
    for (j = 0; j < n; j++)
        acc += wts[x * kmax + j] * row[j];
    out[(c * h + y) * wout + x] = acc;
}

/* Along y: (c, hin, w) -> (c, hout, w). Global size (w, hout, c). */
__kernel void resize_y(__global const float *in, __global float *out, int hin,
                       int hout, int w, __global const int *idx,
                       __global const float *wts, int kmax)
{
    int x = get_global_id(0), y = get_global_id(1), c = get_global_id(2);
    float acc = 0.0f;
    int j, y0, n;

    if (x >= w || y >= hout)
        return;
    y0 = idx[2 * y];
    n = idx[2 * y + 1];
    for (j = 0; j < n; j++)
        acc += wts[y * kmax + j] * in[(c * hin + y0 + j) * w + x];
    out[(c * hout + y) * w + x] = acc;
}

/* Hard constraint, step 1: d = lr_up - sr as complex numbers. */
__kernel void hc_diff(__global const float *lr_up, __global const float *sr,
                      __global float2 *d, int total)
{
    int i = get_global_id(0);

    if (i < total)
        d[i] = (float2)(lr_up[i] - sr[i], 0.0f);
}

/* Hard constraint, step 3: multiply the spectrum of every channel by the
 * low-pass mask, already moved to unshifted frequency order. */
__kernel void hc_mask(__global float2 *d, __global const float *mask,
                      int plane, int total)
{
    int i = get_global_id(0);

    if (i < total)
        d[i] *= mask[i % plane];
}

/* Hard constraint, step 5: sr += Re(ifft) / n^2. */
__kernel void hc_add(__global float *sr, __global const float2 *d,
                     float norm, int total)
{
    int i = get_global_id(0);

    if (i < total)
        sr[i] += d[i].x * norm;
}

#define FFT_MAX 1024

/* In-place radix-2 FFT of power-of-two lines of length n (n <= FFT_MAX),
 * one line per work-group, held in local memory. Line l of channel c
 * starts at c * n * n + l * line_stride and its elements are elem_stride
 * apart, so rows use (n, 1) and columns (1, n). sign is -1 for the
 * forward and +1 for the (unnormalised) inverse transform. Global size
 * n_lines * local size, n_lines = channels * n. */
__kernel void fft_lines(__global float2 *data, int n, int logn,
                        int line_stride, int elem_stride, float sign)
{
    __local float2 buf[FFT_MAX];
    int line = get_group_id(0), lid = get_local_id(0);
    int lsz = get_local_size(0);
    int c = line / n, l = line % n;
    __global float2 *base = data + c * n * n + l * line_stride;
    int i, s;

    /* Load in bit-reversed order. */
    for (i = lid; i < n; i += lsz) {
        int r = 0, v = i, b;

        for (b = 0; b < logn; b++) {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        buf[r] = base[i * elem_stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (s = 1; s <= logn; s++) {
        int span = 1 << (s - 1), m = span << 1;
        int b;

        for (b = lid; b < n / 2; b += lsz) {
            int kk = b & (span - 1);
            int p = (b >> (s - 1)) * m + kk, q = p + span;
            float cs, sn = sincos(sign * 2.0f * M_PI_F * kk / m, &cs);
            float2 u = buf[p], v = buf[q], t;

            t = (float2)(v.x * cs - v.y * sn, v.x * sn + v.y * cs);
            buf[p] = u + t;
            buf[q] = u - t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (i = lid; i < n; i += lsz)
        base[i * elem_stride] = buf[i];
}

/* ------------------------------------------------------------------ */
/* Full SEN2SR pipeline kernels (MambaSR, Swin2SR, hard constraints). */

/* k x k convolution (k = 1 or 3), stride 1, zero padding k / 2, with a
 * fused epilogue:
 *   ACT_SILU  y = x * sigmoid(x)
 *   ACT_ATTN  y = (x + res) * (sigmoid(x) - 0.5), the SPAB attention
 *   ACT_RELU  y = max(x, 0), the SEN2SR positivity clamp
 * With shuffle r > 1 the result is written through a pixel shuffle:
 * channel co goes to plane co / r^2 at sub-pixel ((co % r^2) / r,
 * co % r) of an (h * r, w * r) output.
 * Weights are (cout, cin, k, k), bias (cout). in_h, out_h and res_h say
 * whether in, out and res hold half values. Global size
 * (w, h, ceil(cout / COB)). */
__kernel void conv2d_full(__global const void *in, int in_off, int in_h, int cin,
                     __global const float *wt, __global const float *bias,
                     __global void *out, int out_off, int out_h, int cout,
                     int h, int w, int k, int act, int shuffle,
                     __global const void *res, int res_off, int res_h, int base)
{
    int id = FLAT_ID, x, y, co0;
    int plane = h * w, pad = k / 2, kk = k * k;
    float acc[COB];
    int j, ci, dy, dx;

    if (id >= w * h * ((cout + COB - 1) / COB))
        return;
    x = id % w;
    id /= w;
    y = id % h;
    co0 = (id / h) * COB;
    for (j = 0; j < COB; j++)
        acc[j] = co0 + j < cout ? bias[co0 + j] : 0.0f;

    for (ci = 0; ci < cin; ci++) {
        int src = in_off + ci * plane;

        for (dy = 0; dy < k; dy++) {
            int yy = y + dy - pad;

            if (yy < 0 || yy >= h)
                continue;
            for (dx = 0; dx < k; dx++) {
                int xx = x + dx - pad;
                float v;

                if (xx < 0 || xx >= w)
                    continue;
                v = ld(in, src + yy * w + xx, in_h);
                for (j = 0; j < COB; j++)
                    if (co0 + j < cout)
                        acc[j] +=
                            v * wt[((co0 + j) * cin + ci) * kk + dy * k + dx];
            }
        }
    }

    for (j = 0; j < COB && co0 + j < cout; j++)
        conv_store(out, out_off, out_h, res, res_off, res_h, co0 + j, x, y, h,
                   w, act, shuffle, acc[j]);
}

/* conv2d_full for CPU devices (profile=cpu): same contract, but each
 * work-item computes 8 consecutive pixels of COB output channels with
 * 8-wide vector loads along the row, which PoCL maps onto the SIMD
 * units. Global size (ceil(w / 8), h, ceil(cout / COB)). */
__kernel void conv2d_cpu_full(__global const void *in, int in_off, int in_h,
                         int cin, __global const float *wt,
                         __global const float *bias, __global void *out,
                         int out_off, int out_h, int cout, int h, int w, int k,
                         int act, int shuffle, __global const void *res,
                         int res_off, int res_h, int base)
{
    int id = FLAT_ID, x0, y, co0, xb = (w + 7) / 8;
    int plane = h * w, pad = k / 2, kk = k * k;
    float8 acc[COB];
    int j, ci, dy, dx, i;

    if (id >= xb * h * ((cout + COB - 1) / COB))
        return;
    x0 = (id % xb) * 8;
    id /= xb;
    y = id % h;
    co0 = (id / h) * COB;
    for (j = 0; j < COB; j++)
        acc[j] = (float8)(co0 + j < cout ? bias[co0 + j] : 0.0f);

    for (ci = 0; ci < cin; ci++) {
        int src = in_off + ci * plane;

        for (dy = 0; dy < k; dy++) {
            int yy = y + dy - pad;

            if (yy < 0 || yy >= h)
                continue;
            for (dx = 0; dx < k; dx++) {
                const float8 v = ld_row8(in, src + yy * w, x0 + dx - pad, w,
                                         in_h);

                for (j = 0; j < COB; j++)
                    if (co0 + j < cout)
                        acc[j] = mad(
                            (float8)(wt[((co0 + j) * cin + ci) * kk + dy * k +
                                        dx]),
                            v, acc[j]);
            }
        }
    }

    for (j = 0; j < COB && co0 + j < cout; j++) {
        const float a[8] = {acc[j].s0, acc[j].s1, acc[j].s2, acc[j].s3,
                            acc[j].s4, acc[j].s5, acc[j].s6, acc[j].s7};

        for (i = 0; i < 8 && x0 + i < w; i++)
            conv_store(out, out_off, out_h, res, res_off, res_h, co0 + j,
                       x0 + i, y, h, w, act, shuffle, a[i]);
    }
}

/* Nearest-neighbour 2x decimation keeping even rows and columns, as
 * torch interpolate(scale_factor=0.5, mode="nearest"). Global size
 * (w / 2, h / 2, c). */
__kernel void decimate2_full(__global const float *in, __global float *out, int h,
                        int w, int c_count, int base)
{
    int id = FLAT_ID, x, y, c;
    int ho = h / 2, wo = w / 2;

    if (id >= ho * wo * c_count)
        return;
    x = id % wo;
    id /= wo;
    y = id % ho;
    c = id / ho;
    out[(c * ho + y) * wo + x] = in[(c * h + 2 * y) * w + 2 * x];
}

/* Separable anti-aliased resampling, one axis at a time, with the
 * per-output-index support start/length (idx) and normalised weights
 * (wts, kmax per index) precomputed on the host as in PyTorch's
 * antialias=True path. */

/* Along x: (c, h, win) -> (c, h, wout). Global size (wout, h, c). */
__kernel void resize_x_full(__global const float *in, __global float *out, int h,
                       int win, int wout, int C, __global const int *idx,
                       __global const float *wts, int kmax, int base)
{
    int id = FLAT_ID, x, y, c;
    __global const float *row;
    float acc = 0.0f;
    int j, x0, n;

    if (id >= wout * h * C)
        return;
    x = id % wout;
    id /= wout;
    y = id % h;
    c = id / h;
    x0 = idx[2 * x];
    n = idx[2 * x + 1];
    row = in + (c * h + y) * win + x0;
    for (j = 0; j < n; j++)
        acc += wts[x * kmax + j] * row[j];
    out[(c * h + y) * wout + x] = acc;
}

/* Along y: (c, hin, w) -> (c, hout, w). Global size (w, hout, c). */
__kernel void resize_y_full(__global const float *in, __global float *out, int hin,
                       int hout, int w, int C, __global const int *idx,
                       __global const float *wts, int kmax, int base)
{
    int id = FLAT_ID, x, y, c;
    float acc = 0.0f;
    int j, y0, n;

    if (id >= w * hout * C)
        return;
    x = id % w;
    id /= w;
    y = id % hout;
    c = id / hout;
    y0 = idx[2 * y];
    n = idx[2 * y + 1];
    for (j = 0; j < n; j++)
        acc += wts[y * kmax + j] * in[(c * hin + y0 + j) * w + x];
    out[(c * hout + y) * w + x] = acc;
}

/* Hard constraint, step 1: d = lr_up - sr as complex numbers. */
__kernel void hc_diff_full(__global const float *lr_up, __global const float *sr,
                      __global float2 *d, int total, int base)
{
    int i = FLAT_ID;

    if (i < total)
        d[i] = (float2)(lr_up[i] - sr[i], 0.0f);
}

/* Hard constraint, step 3: multiply the spectrum of every channel by the
 * low-pass mask, already moved to unshifted frequency order. */
__kernel void hc_mask_full(__global float2 *d, __global const float *mask,
                      int plane, int total, int base)
{
    int i = FLAT_ID;

    if (i < total)
        d[i] *= mask[i % plane];
}

/* Hard constraint, step 5: sr += Re(ifft) / n^2. */
__kernel void hc_add_full(__global float *sr, __global const float2 *d,
                     float norm, int total, int base)
{
    int i = FLAT_ID;

    if (i < total)
        sr[i] += d[i].x * norm;
}


/* In-place radix-2 FFT of power-of-two lines of length n (n <= FFT_MAX),
 * one line per work-group, held in local memory. Line l of channel c
 * starts at c * n * n + l * line_stride and its elements are elem_stride
 * apart, so rows use (n, 1) and columns (1, n). sign is -1 for the
 * forward and +1 for the (unnormalised) inverse transform. Global size
 * n_lines * local size, n_lines = channels * n. */
__kernel void fft_lines_full(__global float2 *data, int n, int logn,
                        int line_stride, int elem_stride, float sign)
{
    __local float2 buf[FFT_MAX];
    int line = get_group_id(1) + get_group_id(2) * get_num_groups(1);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int c = line / n, l = line % n;
    __global float2 *base = data + c * n * n + l * line_stride;
    int i, s;

    /* Load in bit-reversed order. */
    for (i = lid; i < n; i += lsz) {
        int r = 0, v = i, b;

        for (b = 0; b < logn; b++) {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        buf[r] = base[i * elem_stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (s = 1; s <= logn; s++) {
        int span = 1 << (s - 1), m = span << 1;
        int b;

        for (b = lid; b < n / 2; b += lsz) {
            int kk = b & (span - 1);
            int p = (b >> (s - 1)) * m + kk, q = p + span;
            float cs, sn = sincos(sign * 2.0f * M_PI_F * kk / m, &cs);
            float2 u = buf[p], v = buf[q], t;

            t = (float2)(v.x * cs - v.y * sn, v.x * sn + v.y * cs);
            buf[p] = u + t;
            buf[q] = u - t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (i = lid; i < n; i += lsz)
        base[i * elem_stride] = buf[i];
}

/* Additional kernels for the full SEN2SR models (MambaSR + Swin2SR).
 * All tensors are float32 NCHW unless noted. */

/* C[M,N] = A[M,K] * B[N,K]^T + bias[N].  PyTorch Linear stores its
 * weight as (out, in), hence the transposed access to B. */
__kernel void gemm(__global const float *a, __global const float *b,
                   __global const float *bias, int has_bias,
                   __global float *c, int M, int K, int N, int row_base, int base)
{
    int id = FLAT_ID, m, n, k;
    float s;

    if (id >= M * N)
        return;
    m = row_base + id / N;
    n = id % N;
    s = has_bias ? bias[n] : 0.0f;
    for (k = 0; k < K; k++)
        s += a[(long)m * K + k] * b[(long)n * K + k];
    c[(long)m * N + n] = s;
}

/* LayerNorm over the last dimension of an M x C matrix. */
__kernel void layernorm(__global const float *x, __global float *y,
                        __global const float *g, __global const float *b,
                        int M, int C, float eps, int base)
{
    int m = FLAT_ID, i;
    float mean = 0.0f, var = 0.0f, scale;

    if (m >= M)
        return;
    for (i = 0; i < C; i++)
        mean += x[(long)m * C + i];
    mean /= C;
    for (i = 0; i < C; i++) {
        float d = x[(long)m * C + i] - mean;
        var += d * d;
    }
    var /= C;
    scale = 1.0f / sqrt(var + eps);
    for (i = 0; i < C; i++)
        y[(long)m * C + i] =
            (x[(long)m * C + i] - mean) * scale * g[i] + b[i];
}

/* Element-wise activation.  Codes match ACT_* in sen2sr_opencl.cl. */
__kernel void act_k(__global const float *x, __global float *y, int n,
                    int code, float p, int base)
{
    int i = FLAT_ID;
    float v;

    if (i >= n)
        return;
    v = x[i];
    if (code == 1)
        v = v / (1.0f + exp(-v));                 /* SiLU */
    else if (code == 3)
        v = fmax(v, 0.0f);                        /* ReLU */
    else if (code == 4)
        v = 0.5f * v * (1.0f + erf(v * 0.7071067811865476f)); /* GELU */
    else if (code == 5)
        v = v >= 0.0f ? v : p * v;                /* LeakyReLU */
    else if (code == 6)
        v = 1.0f / (1.0f + exp(-v));              /* Sigmoid */
    y[i] = v;
}

/* y[i] = x[map[i]] -- window partition, cyclic roll, transpose and
 * flip are all expressed as gather maps built on the host. */
__kernel void gather_k(__global const float *x, __global float *y,
                       __global const int *map, int n, int base)
{
    int i = FLAT_ID;
    if (i < n)
        y[i] = x[map[i]];
}

/* y[i] = a[i] * sa + b[i] * sb + bias, for broadcast elementwise ops. */
__kernel void lincomb(__global const float *a, __global const float *b,
                      __global float *y, int n, float sa, float sb, float bias, int base)
{
    int i = FLAT_ID;
    if (i < n)
        y[i] = a[i] * sa + b[i] * sb + bias;
}

/* Row-wise softmax, one work-item per row of N elements. */
__kernel void softmax_k(__global const float *x, __global float *y,
                        int rows, int N, int base)
{
    int r = FLAT_ID, i;
    float mx = -1.0e30f, sum = 0.0f;

    if (r >= rows)
        return;
    for (i = 0; i < N; i++)
        mx = fmax(mx, x[(long)r * N + i]);
    for (i = 0; i < N; i++)
        sum += exp(x[(long)r * N + i] - mx);
    for (i = 0; i < N; i++)
        y[(long)r * N + i] = exp(x[(long)r * N + i] - mx) / sum;
}

/* Depthwise 3x3 convolution, zero padded, C independent channels.
 * Weight layout is (C, 3, 3), bias (C). */
__kernel void conv_dw3(__global const float *in, __global const float *wt,
                       __global const float *bias, __global float *out,
                       int C, int H, int W, int base)
{
    int id = FLAT_ID, x, y, c;
    int plane = H * W, dy, dx;
    float s;

    if (id >= C * H * W)
        return;
    x = id % W;
    id /= W;
    y = id % H;
    c = id / H;
    s = bias[c];
    for (dy = 0; dy < 3; dy++) {
        int yy = y + dy - 1;
        if (yy < 0 || yy >= H)
            continue;
        for (dx = 0; dx < 3; dx++) {
            int xx = x + dx - 1;
            if (xx < 0 || xx >= W)
                continue;
            s += in[c * plane + yy * W + xx] * wt[c * 9 + dy * 3 + dx];
        }
    }
    out[c * plane + y * W + x] = s;
}

/* Global average pool over H x W, one output per channel. */
__kernel void pool1(__global const float *x, __global float *y, int C, int HW, int base)
{
    int c = FLAT_ID, i;
    float s = 0.0f;
    if (c >= C)
        return;
    for (i = 0; i < HW; i++)
        s += x[(long)c * HW + i];
    y[c] = s / HW;
}

/* Selective scan used by SS2D.  One work-item per (direction, channel);
 * the recurrence over the L tokens is sequential, state size N <= 16.
 * A is already -exp(A_logs), dts are pre-biased, softplus is applied here. */
__kernel void selective_scan(__global const float *xs, __global const float *dts,
                             __global const float *A, __global const float *B,
                             __global const float *C, __global const float *D,
                             __global const float *dt_bias,
                             __global float *y, int Dinner, int N, int L, int base)
{
    int row = FLAT_ID, rows = Dinner * 4;
    int k, n, l;
    float state[16];

    if (row >= rows)
        return;
    k = row / Dinner;
    for (n = 0; n < 16; n++)
        state[n] = 0.0f;
    for (l = 0; l < L; l++) {
        float v = dts[(long)row * L + l] + dt_bias[row];
        float dt = fmax(v, 0.0f) + log(1.0f + exp(-fabs(v)));
        float x = xs[(long)row * L + l];
        float acc = 0.0f;
        for (n = 0; n < N; n++) {
            float b = B[((long)k * N + n) * L + l];
            float c = C[((long)k * N + n) * L + l];
            float e = exp(dt * A[(long)row * N + n]);
            state[n] = e * state[n] + dt * b * x;
            acc += state[n] * c;
        }
        y[(long)row * L + l] = acc + D[row] * x;
    }
}

/* Select channels from an NCHW tensor: out[k,i] = in[idx[k],i]. */
__kernel void gather_bands(__global const float *in, __global float *out,
                           __global const int *idx, int HW, int K, int base)
{
    int id = FLAT_ID, i, k;
    if (id >= HW * K)
        return;
    i = id % HW;
    k = id / HW;
    out[(long)k * HW + i] = in[(long)idx[k] * HW + i];
}

/* Concatenate two NCHW tensors along channels: first CA planes of a,
 * then K planes of b. */
__kernel void concat2(__global const float *a, __global const float *b,
                      __global float *out, int CA, int K, int HW, int base)
{
    int id = FLAT_ID, c, i;
    if (id >= HW * (CA + K))
        return;
    c = id / HW;
    i = id % HW;
    out[(long)c * HW + i] = c < CA ? a[(long)c * HW + i]
                                   : b[(long)(c - CA) * HW + i];
}

/* Clamp negative values to zero. */
__kernel void clamp_pos(__global float *x, int n, int base)
{
    int i = FLAT_ID;
    if (i < n && x[i] < 0.0f)
        x[i] = 0.0f;
}

/* Layout conversion between NCHW feature maps [C][H*W] and the token
 * matrices [H*W][C] used by Linear/LayerNorm. */
__kernel void nchw2tokens(__global const float *x, __global float *y,
                          int C, int L, int base)
{
    int id = FLAT_ID, t, c;
    if (id >= L * C)
        return;
    t = id / C;
    c = id % C;
    y[(long)t * C + c] = x[(long)c * L + t];
}

__kernel void tokens2nchw(__global const float *x, __global float *y,
                          int C, int L, int base)
{
    int id = FLAT_ID, c, t;
    if (id >= L * C)
        return;
    c = id / L;
    t = id % L;
    y[(long)c * L + t] = x[(long)t * C + c];
}

/* Split a fused qkv matrix into the Swin2SR (window, head, token, dim)
 * layout used by the attention kernels.  qkv is (nW*N, 3*C). */
__kernel void qkv_split(__global const float *qkv, __global float *q,
                        __global float *k, __global float *v, int nW,
                        int heads, int N, int C, int base)
{
    int id = FLAT_ID, w, h, i, z, d = C / heads;
    long qo, so;

    if (id >= nW * heads * N)
        return;
    i = id % N;
    id /= N;
    h = id % heads;
    w = id / heads;
    qo = ((long)w * heads + h) * N * d + i * d;
    so = ((long)w * N + i) * 3 * C + h * d;
    for (z = 0; z < d; z++) {
        q[qo + z] = qkv[so + z];
        k[qo + z] = qkv[so + C + z];
        v[qo + z] = qkv[so + 2 * C + z];
    }
}

/* Inverse of qkv_split: concatenate heads back to (window, token, C). */
__kernel void head_concat(__global const float *x, __global float *y,
                          int nW, int heads, int N, int C, int base)
{
    int id = FLAT_ID, w, h, i, z, d = C / heads;
    long si, so;

    if (id >= nW * heads * N)
        return;
    i = id % N;
    id /= N;
    h = id % heads;
    w = id / heads;
    si = ((long)w * heads + h) * N * d + i * d;
    so = ((long)w * N + i) * C + h * d;
    for (z = 0; z < d; z++)
        y[so + z] = x[si + z];
}

/* y[c,i] = x[c,i] * s[c], for NCHW planes with C outer. */
__kernel void mul_bcast(__global const float *x, __global const float *s,
                        __global float *y, int C, int HW, int base)
{
    int id = FLAT_ID, c, i;
    if (id >= HW * C)
        return;
    c = id / HW;
    i = id % HW;
    y[(long)c * HW + i] = x[(long)c * HW + i] * s[c];
}

/* y[:,0:D] = x[:,0:D], y[:,D:2D] = x[:,D:2D]. */
__kernel void split2(__global const float *x, __global float *a,
                     __global float *b, int rows, int D, int base)
{
    int id = FLAT_ID, r, i, z;
    if (id >= rows * D)
        return;
    r = id / D;
    i = id % D;
    a[(long)r * D + i] = x[(long)r * 2 * D + i];
    b[(long)r * D + i] = x[(long)r * 2 * D + D + i];
}

/* x_dbl[k,c,l] = sum_d xs[k,d,l] * wt[k,c,d].
 * wt is (K, 38, Dinner), xs is (K*Dinner, L), c < 38. */
__kernel void proj_xdbl(__global const float *xs, __global const float *wt,
                        __global float *out, int K, int Dinner, int C, int L, int base)
{
    int id = FLAT_ID, k, c, l, d;
    float s = 0.0f;

    if (id >= K * C * L)
        return;
    l = id % L;
    id /= L;
    c = id % C;
    k = id / C;
    for (d = 0; d < Dinner; d++)
        s += xs[((long)k * Dinner + d) * L + l] *
             wt[((long)k * C + c) * Dinner + d];
    out[((long)k * C + c) * L + l] = s;
}

/* dts[k,d,l] = sum_r x[k,r,l] * wt[k,d,r], r < dt_rank.
 * wt is (K, Dinner, rank), x is (K*rank, L). */
__kernel void proj_dts(__global const float *x, __global const float *wt,
                       __global float *out, int K, int Dinner, int R, int L, int base)
{
    int id = FLAT_ID, k, d, l, r;
    float s = 0.0f;

    if (id >= K * Dinner * L)
        return;
    l = id % L;
    id /= L;
    d = id % Dinner;
    k = id / Dinner;
    for (r = 0; r < R; r++)
        s += x[((long)k * R + r) * L + l] *
             wt[((long)k * Dinner + d) * R + r];
    out[((long)k * Dinner + d) * L + l] = s;
}

/* Build the four SS2D scan directions from an NCHW feature map.  src is a
 * channel of D x H x W stored as [D][H*W]. */
__kernel void make_xs(__global const float *x, __global float *xs,
                      int D, int H, int W, int base)
{
    int id = FLAT_ID, t, d, k;
    int yy, xx, L = H * W;
    int src;

    if (id >= 4 * D * L)
        return;
    t = id % L;
    id /= L;
    d = id % D;
    k = id / D;
    yy = t / W;
    xx = t % W;
    if (k == 0)
        src = t;
    else if (k == 1)
        src = xx * H + yy;
    else if (k == 2)
        src = L - 1 - t;
    else
        src = L - 1 - (xx * H + yy);
    xs[((long)k * D + d) * L + t] = x[(long)d * L + src];
}

/* Split x_dbl (K, rank+2N, L) into dts (K*rank, L), B and C
 * (K*N, L). */
__kernel void split_xdbl(__global const float *x, __global float *dts,
                         __global float *b, __global float *c,
                         int K, int R, int N, int L, int base)
{
    int id = FLAT_ID, k, i, t;
    long src;

    if (id >= K * (R + 2 * N) * L)
        return;
    t = id % L;
    id /= L;
    i = id % (R + 2 * N);
    k = id / (R + 2 * N);
    src = ((long)k * (R + 2 * N) + i) * L + t;
    if (i < R)
        dts[((long)k * R + i) * L + t] = x[src];
    else if (i < R + N)
        b[((long)k * N + i - R) * L + t] = x[src];
    else
        c[((long)k * N + i - R - N) * L + t] = x[src];
}

/* Token-major broadcast: y[t*C+c] = x[t*C+c] * s[c]. */
__kernel void mul_bcast_last(__global const float *x, __global const float *s,
                             __global float *y, int C, int L, int base)
{
    int id = FLAT_ID, t, c;
    if (id >= L * C)
        return;
    t = id / C;
    c = id % C;
    y[(long)t * C + c] = x[(long)t * C + c] * s[c];
}

/* Four-direction recombination of the SS2D scan outputs.  The row-major
 * scan orders are (y,x), flip(y,x), (x,y) and reversed (x,y). */
__kernel void combine_y(__global const float *y0, __global float *y,
                        int Dinner, int H, int W, int base)
{
    int id = FLAT_ID, x, yc, d;
    int L = H * W;
    int i;

    if (id >= Dinner * H * W)
        return;
    x = id % W;
    id /= W;
    yc = id % H;
    d = id / H;
    i = yc * W + x;
    y[(long)i * Dinner + d] =
        y0[(long)d * L + i] +
        y0[(long)(Dinner + d) * L + (L - 1 - i)] +
        y0[(long)(2 * Dinner + d) * L + x * H + yc] +
        y0[(long)(3 * Dinner + d) * L + (L - 1 - (x * H + yc))];
}

/* Cosine attention logits: normalize q and k per head and token, then
 * dot-product.  q, k are (nW, heads, N, d), attn is (nW, heads, N, N). */
__kernel void attn_qk(__global const float *q, __global const float *k,
                      __global const float *scale, __global float *attn,
                      int nW, int heads, int N, int d, int base)
{
    int id = FLAT_ID, w, h, i, j, z;
    float qn[32], kn[32], nq = 0.0f, nk, s;

    if (id >= nW * heads * N)
        return;
    i = id % N;
    id /= N;
    h = id % heads;
    w = id / heads;
    for (z = 0; z < d; z++) {
        qn[z] = q[((long)w * heads + h) * N * d + i * d + z];
        nq += qn[z] * qn[z];
    }
    nq = 1.0f / sqrt(nq + 1e-12f);
    for (j = 0; j < N; j++) {
        nk = 0.0f;
        for (z = 0; z < d; z++) {
            kn[z] = k[((long)w * heads + h) * N * d + j * d + z];
            nk += kn[z] * kn[z];
        }
        nk = 1.0f / sqrt(nk + 1e-12f);
        s = 0.0f;
        for (z = 0; z < d; z++)
            s += qn[z] * nq * kn[z] * nk;
        attn[((long)w * heads + h) * N * N + i * N + j] = s * scale[h];
    }
}

/* attn += relative_position_bias[h,i,j] + attn_mask[w,i,j].
 * One work-item owns (head,i,j) and loops over windows, so the launch stays
 * small even when the attention tensor has hundreds of millions of entries. */
__kernel void attn_add_bias(__global float *attn, __global const float *rel,
                            __global const float *mask, int nW, int heads,
                            int N, int total)
{
    int j = get_global_id(0), i = get_global_id(1), h = get_global_id(2);
    int plane = N * N, w;
    long r = (long)h * plane + i * N + j;

    (void)total;
    if (j >= N || i >= N || h >= heads)
        return;
    for (w = 0; w < nW; w++) {
        long idx = ((long)w * heads + h) * plane + r;
        attn[idx] += rel[r] + mask[(long)w * plane + i * N + j];
    }
}

/* Context vectors: out[w,h,i,d] = sum_j attn[w,h,i,j] * v[w,h,j,d]. */
__kernel void attn_av(__global const float *attn, __global const float *v,
                      __global float *out, int nW, int heads, int N, int d,
                      int base)
{
    int id = FLAT_ID, w, h, i, j, z;
    if (id >= nW * heads * N)
        return;
    i = id % N;
    id /= N;
    h = id % heads;
    w = id / heads;
    for (z = 0; z < d; z++) {
        float s = 0.0f;
        for (j = 0; j < N; j++)
            s += attn[((long)w * heads + h) * N * N + i * N + j] *
                 v[((long)w * heads + h) * N * d + j * d + z];
        out[((long)w * heads + h) * N * d + i * d + z] = s;
    }
}

/* Anti-aliased resize through four precomputed neighbours per output
 * element.  idx has OH*OW*4 pixel indices, wts the matching weights. */
__kernel void resize4(__global const float *x, __global float *y,
                      __global const int *idx, __global const float *wts,
                      int C, int H, int W, int OH, int OW, int base)
{
    int id = FLAT_ID, c, ox, oy;
    long o;
    float s = 0.0f;
    int j;

    if (id >= C * OH * OW)
        return;
    ox = id % OW;
    id /= OW;
    oy = id % OH;
    c = id / OH;
    o = (long)oy * OW + ox;
    for (j = 0; j < 4; j++)
        s += x[(long)c * H * W + idx[o * 4 + j]] * wts[o * 4 + j];
    y[(long)c * OH * OW + o] = s;
}
