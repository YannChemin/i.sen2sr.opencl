/****************************************************************************
 *
 * MODULE:       i.sen2sr.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Select, discover and download SEN2SR model weights, and
 *               dispatch tiles to the SEN2SRLite or full SEN2SR engine.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

#include "sen2sr.h"
#include "safetensors.h"
#include "util.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define HF_BASE "https://huggingface.co/tacofoundation/SEN2SR/resolve/main"

static const char *full_files[] = {
    "sr_model.safetensor",   "sr_hard_constraint.safetensor",
    "f2_model.safetensor",   "f2_hard_constraint.safetensor",
    "model.safetensor",      "hard_constraint.safetensor"};
static const char *main_files[] = {
    "model.safetensor",      "hard_constraint.safetensor",
    "sr_model.safetensor",   "sr_hard_constraint.safetensor",
    "f2_model.safetensor",   "f2_hard_constraint.safetensor"};
static const char *small_files[] = {"model.safetensor",
                                    "hard_constraint.safetensor"};

struct model_spec {
    const char *variant;
    const char *rel[3];
    const char *hf_dir;
    const char **files;
    int nfiles;
};

static const struct model_spec *spec_for(const char *variant)
{
    static const struct model_spec specs[] = {
        {"full", {"SEN2SR/SEN2SR/main", NULL}, "SEN2SR/main", full_files, 6},
        {"main",
         {"SEN2SR/SEN2SRLite/main", "SEN2SRLite/main", NULL},
         "SEN2SRLite/main",
         main_files,
         6},
        {"rgbn_x4",
         {"SEN2SR/SEN2SRLite/NonReference_RGBN_x4",
          "SEN2SRLite/NonReference_RGBN_x4",
          NULL},
         "SEN2SRLite/NonReference_RGBN_x4",
         small_files,
         2},
        {"rswir_x2",
         {"SEN2SR/SEN2SRLite/Reference_RSWIR_x2",
          "SEN2SRLite/Reference_RSWIR_x2",
          NULL},
         "SEN2SRLite/Reference_RSWIR_x2",
         small_files,
         2}};
    int i;

    for (i = 0; i < (int)(sizeof(specs) / sizeof(specs[0])); i++)
        if (strcmp(specs[i].variant, variant) == 0)
            return &specs[i];
    return NULL;
}

static int have(const char *dir, const char *name, char *path, size_t size)
{
    snprintf(path, size, "%s/%s", dir, name);
    return access(path, R_OK) == 0;
}

static int is_lite(const char *dir)
{
    char path[GPATH_MAX];
    struct st_file f;
    int found;

    if (!have(dir, "model.safetensor", path, sizeof(path)))
        return 0;
    st_open(&f, path);
    found = st_find(&f, "conv_1.conv.0.weight") != NULL;
    st_close(&f);
    return found;
}

static int bundle_ok(const struct model_spec *spec, const char *dir)
{
    char path[GPATH_MAX];
    int i;

    if (strcmp(spec->variant, "full") == 0)
        return sr_full_detect(dir);
    for (i = 0; i < spec->nfiles; i++)
        if (!have(dir, spec->files[i], path, sizeof(path)))
            return 0;
    /* The full and Lite main bundles use the same file names. */
    if (strcmp(spec->variant, "main") == 0 && !is_lite(dir))
        return 0;
    return 1;
}

int sen2sr_default_model(const char *root, const char *variant, char *path,
                         size_t size)
{
    const struct model_spec *spec = spec_for(variant);
    char dir[GPATH_MAX];
    int i;

    if (!spec)
        return 0;
    for (i = 0; i < 2 && spec->rel[i]; i++) {
        snprintf(dir, sizeof(dir), "%s/%s", root, spec->rel[i]);
        if (bundle_ok(spec, dir)) {
            snprintf(path, size, "%s", dir);
            return 1;
        }
    }
    return 0;
}

/* Shell-quote a path so that weights under a HOME with spaces still work. */
static char *quote(const char *s)
{
    size_t n = strlen(s), i, j = 0;
    char *q = G_malloc(2 * n + 3);

    q[j++] = '\'';
    for (i = 0; i < n; i++) {
        if (s[i] == '\'') {
            q[j++] = '\'';
            q[j++] = '\\';
            q[j++] = '\'';
        }
        q[j++] = s[i];
    }
    q[j++] = '\'';
    q[j] = '\0';
    return q;
}

static int run_shell(const char *fmt, ...)
{
    char cmd[16384];
    va_list ap;
    int status;

    va_start(ap, fmt);
    if (vsnprintf(cmd, sizeof(cmd), fmt, ap) >= (int)sizeof(cmd)) {
        va_end(ap);
        G_fatal_error(_("Download command is too long"));
    }
    va_end(ap);
    G_debug(1, "running: %s", cmd);
    status = system(cmd);
    return status == 0;
}

static int download_file(const char *url, const char *dst)
{
    char part[GPATH_MAX];
    char *qurl = quote(url), *qdst = quote(dst), *qpart;
    int ok;

    snprintf(part, sizeof(part), "%s.part", dst);
    qpart = quote(part);
    G_message(_("Downloading %s"), url);
    if (system("command -v curl >/dev/null 2>&1") == 0)
        ok = run_shell("curl -fL --retry 3 --connect-timeout 30 -o %s %s && "
                       "mv -f %s %s",
                       qpart, qurl, qpart, qdst);
    else if (system("command -v wget >/dev/null 2>&1") == 0)
        ok = run_shell("wget -q -O %s %s && mv -f %s %s", qpart, qurl, qpart,
                       qdst);
    else {
        unlink(part);
        G_free(qurl);
        G_free(qdst);
        G_free(qpart);
        G_warning(_("Neither curl nor wget is available to download model "
                    "weights"));
        return 0;
    }
    if (!ok)
        unlink(part);
    G_free(qurl);
    G_free(qdst);
    G_free(qpart);
    return ok;
}

int sen2sr_ensure_default_model(const char *root, const char *variant,
                                char *path, size_t size)
{
    const struct model_spec *spec = spec_for(variant);
    char dir[GPATH_MAX], url[2 * GPATH_MAX], dst[GPATH_MAX], *qdir;
    int i;

    if (sen2sr_default_model(root, variant, path, size))
        return 1;
    if (!spec)
        return 0;

    snprintf(dir, sizeof(dir), "%s/%s", root, spec->rel[0]);
    qdir = quote(dir);
    if (!run_shell("mkdir -p %s", qdir)) {
        G_free(qdir);
        G_warning(_("Cannot create model directory <%s>"), dir);
        return 0;
    }
    G_free(qdir);

    for (i = 0; i < spec->nfiles; i++) {
        snprintf(dst, sizeof(dst), "%s/%s", dir, spec->files[i]);
        if (access(dst, R_OK) == 0)
            continue;
        snprintf(url, sizeof(url), "%s/%s/%s", HF_BASE, spec->hf_dir,
                 spec->files[i]);
        if (!download_file(url, dst))
            return 0;
    }
    if (!bundle_ok(spec, dir)) {
        G_warning(_("Downloaded weights in <%s> are not a valid %s model"), dir,
                  variant);
        return 0;
    }
    snprintf(path, size, "%s", dir);
    return 1;
}

void sen2sr_load(struct sen2sr *s, struct ocl_backend *ocl, const char *dir,
                 const char *variant)
{
    int full = strcmp(variant, "full") == 0;

    memset(s, 0, sizeof(*s));
    if (full) {
        s->engine = SEN2SR_FULL;
        sr_full_load(&s->full, ocl, dir);
        s->scale = s->full.scale;
        s->nin = s->full.nin;
        s->nout = s->full.nout;
        memcpy(s->in_bands, s->full.in_bands, sizeof(s->in_bands));
        memcpy(s->out_bands, s->full.out_bands, sizeof(s->out_bands));
    }
    else {
        s->engine = SEN2SR_LITE;
        sr_model_load(&s->lite, ocl, dir, 0);
        s->scale = s->lite.scale;
        s->nin = s->lite.nin;
        s->nout = s->lite.nout;
        memcpy(s->in_bands, s->lite.in_bands, sizeof(s->in_bands));
        memcpy(s->out_bands, s->lite.out_bands, sizeof(s->out_bands));
    }
}

void sen2sr_run(struct sen2sr *s, const float *in, float *out)
{
    if (s->engine == SEN2SR_FULL)
        sr_full_run(&s->full, in, out);
    else
        sr_model_run(&s->lite, in, out);
}

void sen2sr_free(struct sen2sr *s)
{
    if (s->engine == SEN2SR_FULL)
        sr_full_free(&s->full);
    else
        sr_model_free(&s->lite);
}

const char *sen2sr_name(const struct sen2sr *s)
{
    return s->engine == SEN2SR_FULL ? "SEN2SR main (full Mamba/Swin)"
                                    : sr_variant_name(s->lite.variant);
}
