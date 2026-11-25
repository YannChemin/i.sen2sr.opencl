#include "resize.h"
#include "safetensors.h"
#include "util.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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

static void axis_map(struct ocl_backend *ocl, int in, int out, int cubic,
                     cl_mem *idx, cl_mem *wts, int *kmax)
{
    double scale = (double)in / out;
    double support = (cubic ? 2.0 : 1.0) * (scale >= 1.0 ? scale : 1.0);
    double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    int *ix, i, j, k = 0;
    float *ww;

    ix = malloc((size_t)out * 2 * sizeof(int));
    if (!ix)
        die("out of memory");
    for (i = 0; i < out; i++) {
        double center = scale * (i + 0.5);
        int lo = (int)(center - support + 0.5);
        int hi = (int)(center + support + 0.5);

        if (lo < 0)
            lo = 0;
        if (hi > in)
            hi = in;
        ix[2 * i] = lo;
        ix[2 * i + 1] = hi - lo;
        if (hi - lo > k)
            k = hi - lo;
    }
    ww = calloc((size_t)out * k, sizeof(float));
    if (!ww)
        die("out of memory");
    for (i = 0; i < out; i++) {
        double center = scale * (i + 0.5), total = 0.0, w[64];
        int lo = ix[2 * i], n = ix[2 * i + 1];

        if (n > 64)
            die("resampling support too wide");
        for (j = 0; j < n; j++) {
            w[j] = aa_filter((j + lo - center + 0.5) * invscale, cubic);
            total += w[j];
        }
        for (j = 0; j < n; j++)
            ww[(size_t)i * k + j] = total != 0.0 ? (float)(w[j] / total) : 0.0f;
    }
    *idx = ocl_upload(ocl, ix, (size_t)out * 2 * sizeof(int), "resize idx");
    *wts = ocl_upload(ocl, ww, (size_t)out * k * sizeof(float), "resize wts");
    *kmax = k;
    free(ix);
    free(ww);
}

void resize_map_build(struct ocl_backend *ocl, struct resize_map *r, int in,
                      int out, int cubic)
{
    r->in = in;
    r->out = out;
    axis_map(ocl, in, out, cubic, &r->idx_x, &r->wts_x, &r->kmax);
    axis_map(ocl, in, out, cubic, &r->idx_y, &r->wts_y, &r->kmax);
}

void resize_map_free(struct resize_map *r)
{
    ocl_release(&r->idx_x);
    ocl_release(&r->wts_x);
    ocl_release(&r->idx_y);
    ocl_release(&r->wts_y);
}

cl_mem hc_mask_load(struct ocl_backend *ocl, const char *path, int n)
{
    struct st_file f;
    long shape[2] = {n, n};
    float *mask, *natural;
    cl_mem d;
    int y, x, h = n / 2;

    st_open(&f, path);
    mask = st_get_f32(&f, "weights", 2, shape);
    st_close(&f);
    natural = malloc((size_t)n * n * sizeof(float));
    if (!natural)
        die("out of memory");
    /* The stored mask is in fftshifted (centred) order.  Undo the shift
     * so it matches torch's natural-order FFT output. */
    for (y = 0; y < n; y++)
        for (x = 0; x < n; x++)
            natural[y * n + x] = mask[((y + h) % n) * n + (x + h) % n];
    d = ocl_upload(ocl, natural, (size_t)n * n * sizeof(float), path);
    free(mask);
    free(natural);
    return d;
}
