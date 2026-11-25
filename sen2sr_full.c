/****************************************************************************
 *
 * MODULE:       i.sen2sr.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Full SEN2SR networks (MambaSR and Swin2SR) on an OpenCL
 *               device, behind the tile interface of the GRASS module.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

#include "sen2sr_full.h"
#include "safetensors.h"
#include "util.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const char *required[] = {
    "sr_model.safetensor",   "sr_hard_constraint.safetensor",
    "f2_model.safetensor",   "f2_hard_constraint.safetensor",
    "model.safetensor",      "hard_constraint.safetensor"};

static int have_file(const char *dir, const char *name, char *path, size_t n)
{
    snprintf(path, n, "%s/%s", dir, name);
    return access(path, R_OK) == 0;
}

static int holds_tensor(const char *path, const char *name)
{
    struct st_file f;
    int found;

    st_open(&f, path);
    found = st_find(&f, name) != NULL;
    st_close(&f);
    return found;
}

int sr_full_detect(const char *dir)
{
    char path[GPATH_MAX];
    size_t i;

    if (!have_file(dir, "sr_model.safetensor", path, sizeof(path)))
        return 0;
    if (!holds_tensor(path, "conv_first.weight"))
        return 0;
    if (!have_file(dir, "model.safetensor", path, sizeof(path)))
        return 0;
    if (!holds_tensor(path,
                      "layers.0.residual_group.blocks.0.attn.qkv.weight"))
        return 0;
    for (i = 0; i < sizeof(required) / sizeof(required[0]); i++)
        if (!have_file(dir, required[i], path, sizeof(path)))
            return 0;
    return 1;
}

void sr_full_load(struct sr_full *m, struct ocl_backend *ocl, const char *dir)
{
    static const enum s2_band all[S2_NBANDS] = {B02, B03, B04, B05, B06,
                                                 B07, B08, B8A, B11, B12};

    memset(m, 0, sizeof(*m));
    if (!sr_full_detect(dir))
        G_fatal_error(_("<%s> does not hold the full SEN2SR weights; use a "
                        "SEN2SR/main directory or select a SEN2SRLite model"),
                      dir);

    m->scale = 4;
    m->nin = m->nout = S2_NBANDS;
    memcpy(m->in_bands, all, sizeof(all));
    memcpy(m->out_bands, all, sizeof(all));

    nn_init(&m->nn, ocl);
    pipeline_load(&m->pipe, &m->nn, dir);
    m->in = tensor_alloc(ocl, 1, S2_NBANDS, 128, 128);
    m->out = tensor_alloc(ocl, 1, S2_NBANDS, 512, 512);
}

void sr_full_run(struct sr_full *m, const float *in, float *out)
{
    ocl_write(m->nn.ocl, m->in->d, 0, in, m->in->count * sizeof(float));
    pipeline_forward(&m->pipe, m->in, m->out);
    tensor_read(m->nn.ocl, m->out, out);
}

void sr_full_free(struct sr_full *m)
{
    tensor_free(m->in);
    tensor_free(m->out);
    pipeline_free(&m->pipe);
    nn_free(&m->nn);
}
