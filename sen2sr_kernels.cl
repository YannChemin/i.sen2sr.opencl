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

/* Number of output channels computed per work-item by conv2d. */
#define COB 4

/* k x k convolution (k = 1 or 3), stride 1, zero padding k / 2, with a
 * fused epilogue:
 *   ACT_SILU  y = x * sigmoid(x)
 *   ACT_ATTN  y = (x + res) * (sigmoid(x) - 0.5), the SPAB attention
 *   ACT_RELU  y = max(x, 0), the SEN2SR positivity clamp
 * With shuffle r > 1 the result is written through a pixel shuffle:
 * channel co goes to plane co / r^2 at sub-pixel ((co % r^2) / r,
 * co % r) of an (h * r, w * r) output.
 * Weights are (cout, cin, k, k), bias (cout). Global size
 * (w, h, ceil(cout / COB)). */
__kernel void conv2d(__global const float *in, int in_off, int cin,
                     __global const float *wt, __global const float *bias,
                     __global float *out, int out_off, int cout, int h, int w,
                     int k, int act, int shuffle, __global const float *res,
                     int res_off)
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
        __global const float *src = in + in_off + ci * plane;

        for (dy = 0; dy < k; dy++) {
            int yy = y + dy - pad;

            if (yy < 0 || yy >= h)
                continue;
            for (dx = 0; dx < k; dx++) {
                int xx = x + dx - pad;
                float v;

                if (xx < 0 || xx >= w)
                    continue;
                v = src[yy * w + xx];
                for (j = 0; j < COB; j++)
                    if (co0 + j < cout)
                        acc[j] +=
                            v * wt[((co0 + j) * cin + ci) * kk + dy * k + dx];
            }
        }
    }

    for (j = 0; j < COB; j++) {
        int co = co0 + j;
        float v = acc[j];

        if (co >= cout)
            break;
        if (act == ACT_SILU)
            v = v / (1.0f + exp(-v));
        else if (act == ACT_ATTN)
            v = (v + res[res_off + co * plane + y * w + x]) *
                (1.0f / (1.0f + exp(-v)) - 0.5f);
        else if (act == ACT_RELU)
            v = fmax(v, 0.0f);

        if (shuffle == 1)
            out[out_off + co * plane + y * w + x] = v;
        else {
            int r2 = shuffle * shuffle, c = co / r2, sub = co % r2;
            int oy = y * shuffle + sub / shuffle;
            int ox = x * shuffle + sub % shuffle;

            out[out_off + (c * h * shuffle + oy) * w * shuffle + ox] = v;
        }
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
