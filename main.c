/****************************************************************************
 *
 * MODULE:       i.sen2sr.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Super-resolve Sentinel-2 bands with the SEN2SRLite models
 *               (10 m RGBN and 20 m bands to 2.5 m, or 20 m bands to
 *               10 m) on an OpenCL device.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "ocl_backend.h"
#include "sen2sr_model.h"

static const char *band_suffix[S2_NBANDS] = {"B02", "B03", "B04", "B05", "B06",
                                             "B07", "B08", "B8A", "B11", "B12"};

/* Bundled model directories (see models/README.md), by variant. */
static const struct {
    const char *name;
    enum sr_variant variant;
} bundled[] = {
    {"main", SR_ALL_X4}, {"rgbn_x4", SR_RGBN_X4}, {"rswir_x2", SR_RSWIR_X2}};

/* Path of a bundled model directory installed with the module, searched
 * like G_find_etc() (GRASS_ADDON_ETC, then $GISBASE/etc) and in the
 * etc directory of GRASS_ADDON_BASE, where g.extension installs addon
 * data files. NULL if not found. */
static char *find_bundled_model(const char *name)
{
    char rel[GPATH_MAX], path[GPATH_MAX];
    const char *addon_base = getenv("GRASS_ADDON_BASE");
    char *found;

    snprintf(rel, sizeof(rel), "i.sen2sr.opencl/models/%s", name);
    found = G_find_etc(rel);
    if (found)
        return found;
    if (addon_base) {
        snprintf(path, sizeof(path), "%s/etc/%s", addon_base, rel);
        if (access(path, F_OK) == 0)
            return G_store(path);
    }
    return NULL;
}

/* Tile origins along an axis of padded length len (>= SR_TILE, even), and
 * the first index of each tile's core, where its output is kept: cores
 * meet in the middle of the overlap between neighbouring tiles. Returns
 * the number of tiles; core[n] is set to len. */
static int tile_layout(int len, int overlap, int **start, int **core)
{
    int step = SR_TILE - overlap, n = 0, s = 0, i;

    *start = G_malloc(sizeof(int) * (len / step + 2));
    for (;;) {
        (*start)[n++] = s;
        if (s + SR_TILE >= len)
            break;
        s += step;
        if (s + SR_TILE > len)
            s = len - SR_TILE;
    }
    *core = G_malloc(sizeof(int) * (n + 1));
    (*core)[0] = 0;
    for (i = 1; i < n; i++)
        (*core)[i] = ((*start)[i - 1] + SR_TILE + (*start)[i]) / 2;
    (*core)[n] = len;
    return n;
}

/* Warn when the region is not the 10 m grid the models are trained on,
 * and when a 20 m band is not aligned with the region: the 20 m models
 * rebuild 2 x 2 blocks of the 10 m grid from the region origin. */
static void check_grid(const struct Cell_head *win, const char **names,
                       const enum s2_band *bands, int n)
{
    double f = G_database_units_to_meters_factor();
    int i;

    if (G_projection() == PROJECTION_LL || f <= 0.0)
        G_warning(_("Unable to check that the region resolution is 10 m in "
                    "this coordinate system; the SEN2SR models expect "
                    "Sentinel-2 10 m cells"));
    else if (fabs(win->ns_res * f - 10.0) > 0.01 ||
             fabs(win->ew_res * f - 10.0) > 0.01)
        G_warning(_("The region resolution is %g x %g m, but the SEN2SR "
                    "models expect Sentinel-2 10 m cells"),
                  win->ew_res * f, win->ns_res * f);

    for (i = 0; i < n; i++) {
        struct Cell_head hd;
        double ox, oy;

        if (bands[i] == B02 || bands[i] == B03 || bands[i] == B04 ||
            bands[i] == B08)
            continue;
        Rast_get_cellhd(names[i], "", &hd);
        if (fabs(hd.ew_res - 2.0 * win->ew_res) > 1e-6 * hd.ew_res ||
            fabs(hd.ns_res - 2.0 * win->ns_res) > 1e-6 * hd.ns_res)
            continue;
        ox = (win->west - hd.west) / hd.ew_res;
        oy = (hd.north - win->north) / hd.ns_res;
        if (fabs(ox - floor(ox + 0.5)) > 1e-3 ||
            fabs(oy - floor(oy + 0.5)) > 1e-3)
            G_warning(_("The region is not aligned with the cells of the "
                        "20 m band <%s>; align it with 'g.region align=%s' "
                        "followed by 'g.region res=10' for correct results"),
                      names[i], names[i]);
    }
}

int main(int argc, char *argv[])
{
    struct GModule *module;
    struct Option *band_opt[S2_NBANDS], *output, *model_opt, *model_dir,
        *scale_opt, *offset_opt, *overlap_opt, *device, *platform;
    static const char *band_key[S2_NBANDS] = {
        "blue",     "green", "red",   "rededge1", "rededge2",
        "rededge3", "nir",   "nir08", "swir16",   "swir22"};
    static const char *band_desc[S2_NBANDS] = {
        "Sentinel-2 B02 (blue, 10 m)",
        "Sentinel-2 B03 (green, 10 m)",
        "Sentinel-2 B04 (red, 10 m)",
        "Sentinel-2 B05 (red edge 1, 20 m)",
        "Sentinel-2 B06 (red edge 2, 20 m)",
        "Sentinel-2 B07 (red edge 3, 20 m)",
        "Sentinel-2 B08 (NIR, 10 m)",
        "Sentinel-2 B8A (narrow NIR, 20 m)",
        "Sentinel-2 B11 (SWIR 1.6 um, 20 m)",
        "Sentinel-2 B12 (SWIR 2.2 um, 20 m)"};
    struct ocl_backend ocl;
    struct sr_model model;
    struct Cell_head win, hr;
    const char *in_names[S2_NBANDS];
    char out_names[S2_NBANDS][GNAME_MAX];
    int in_fd[S2_NBANDS], out_fd[S2_NBANDS];
    FCELL *row_buf, *out_row;
    float *strip, *tile, *tile_out, *hr_strip;
    char *null_strip;
    int *ys, *yc, *xs, *xc, nty, ntx, ty, tx;
    int S, T = SR_TILE, H, W, Hp, Wp, nin, nout, overlap, max_core;
    double scale, offset;
    size_t tile_plane, out_plane, hr_cols;
    const char *model_name, *model_path;
    char *desc;
    int i, b, r, c, has_20m;

    G_gisinit(argv[0]);

    module = G_define_module();
    G_add_keyword(_("imagery"));
    G_add_keyword(_("super resolution"));
    G_add_keyword(_("Sentinel"));
    G_add_keyword(_("deep learning"));
    G_add_keyword(_("GPU"));
    G_add_keyword(_("OpenCL"));
    module->description =
        _("Super-resolves Sentinel-2 bands with the SEN2SRLite models on an "
          "OpenCL device.");

    for (b = 0; b < S2_NBANDS; b++) {
        band_opt[b] = G_define_standard_option(G_OPT_R_INPUT);
        band_opt[b]->key = band_key[b];
        band_opt[b]->description = _(band_desc[b]);
        band_opt[b]->required =
            (b == B02 || b == B03 || b == B04 || b == B08) ? YES : NO;
        band_opt[b]->guisection = _("Bands");
    }

    output = G_define_standard_option(G_OPT_R_BASENAME_OUTPUT);
    output->description =
        _("Basename for output raster maps, suffixed with the band name "
          "(e.g. _B02)");

    model_opt = G_define_option();
    model_opt->key = "model";
    model_opt->type = TYPE_STRING;
    model_opt->options = "auto,main,rgbn_x4,rswir_x2";
    model_opt->answer = "auto";
    model_opt->label = _("SEN2SRLite model");
    model_opt->description =
        _("The three models are installed with the module; auto uses main "
          "when 20 m bands are given, else rgbn_x4");
    G_asprintf(&desc, "auto;%s;main;%s;rgbn_x4;%s;rswir_x2;%s",
               _("main if any 20 m band is given, else rgbn_x4"),
               _("all 10 bands to 2.5 m"), _("B02, B03, B04, B08 to 2.5 m"),
               _("20 m bands to 10 m, guided by the 10 m bands"));
    model_opt->descriptions = desc;
    model_opt->guisection = _("Model");

    model_dir = G_define_standard_option(G_OPT_M_DIR);
    model_dir->key = "model_dir";
    model_dir->required = NO;
    model_dir->label = _("Directory of other SEN2SRLite weights");
    model_dir->description =
        _("Use the weights in this directory (one SEN2SRLite variant as "
          "published at huggingface.co/tacofoundation/SEN2SR) instead of "
          "the installed ones");
    model_dir->guisection = _("Model");

    scale_opt = G_define_option();
    scale_opt->key = "scale";
    scale_opt->type = TYPE_DOUBLE;
    scale_opt->answer = "10000";
    scale_opt->label = _("Scale factor of the input values");
    scale_opt->description =
        _("Reflectance = (value + offset) / scale; use 1 for inputs "
          "already in reflectance");
    scale_opt->guisection = _("Input");

    offset_opt = G_define_option();
    offset_opt->key = "offset";
    offset_opt->type = TYPE_DOUBLE;
    offset_opt->answer = "0";
    offset_opt->label = _("Offset added to the input values before scaling");
    offset_opt->description =
        _("E.g. -1000 for L2A products of processing baseline 04.00 or "
          "later; outputs are returned in the input units");
    offset_opt->guisection = _("Input");

    overlap_opt = G_define_option();
    overlap_opt->key = "overlap";
    overlap_opt->type = TYPE_INTEGER;
    overlap_opt->answer = "32";
    overlap_opt->options = "0-96";
    overlap_opt->description =
        _("Overlap between 128 x 128 input tiles in cells (even number)");
    overlap_opt->guisection = _("OpenCL");

    device = G_define_option();
    device->key = "device";
    device->type = TYPE_STRING;
    device->options = "auto,gpu,cpu";
    device->answer = "auto";
    device->description = _("OpenCL device type");
    device->guisection = _("OpenCL");

    platform = G_define_option();
    platform->key = "platform";
    platform->type = TYPE_STRING;
    platform->required = NO;
    platform->description =
        _("Only use OpenCL platforms whose name contains this string "
          "(case-insensitive)");
    platform->guisection = _("OpenCL");

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    scale = atof(scale_opt->answer);
    offset = atof(offset_opt->answer);
    overlap = atoi(overlap_opt->answer);
    if (scale <= 0.0)
        G_fatal_error(_("Option <%s> must be positive"), scale_opt->key);
    if (overlap % 2)
        G_fatal_error(_("Option <%s> must be an even number of cells"),
                      overlap_opt->key);

    /* Model choice: auto picks the model that uses the given bands. The
     * 20 m-to-10 m model is a different product (10 m output) and is only
     * used on request. */
    has_20m = 0;
    for (b = 0; b < S2_NBANDS; b++)
        if (b != B02 && b != B03 && b != B04 && b != B08 && band_opt[b]->answer)
            has_20m = 1;
    model_name = strcmp(model_opt->answer, "auto") != 0 ? model_opt->answer
                 : has_20m                              ? "main"
                                                        : "rgbn_x4";
    if (model_dir->answer)
        model_path = model_dir->answer;
    else {
        model_path = find_bundled_model(model_name);
        if (!model_path)
            G_fatal_error(_("The installed SEN2SRLite model <%s> was not "
                            "found; reinstall the module or give the weights "
                            "with <%s>"),
                          model_name, model_dir->key);
    }

    ocl_init(&ocl, device->answer, platform->answer);
    sr_model_load(&model, &ocl, model_path);
    if (model_dir->answer && strcmp(model_opt->answer, "auto") != 0) {
        for (i = 0; i < (int)(sizeof(bundled) / sizeof(bundled[0])); i++)
            if (strcmp(bundled[i].name, model_opt->answer) == 0 &&
                bundled[i].variant != model.variant)
                G_fatal_error(_("<%s> holds the %s model, not <%s>"),
                              model_dir->answer, sr_variant_name(model.variant),
                              model_opt->answer);
    }
    S = model.scale;
    nin = model.nin;
    nout = model.nout;
    G_message(_("Model: %s, %d input bands, %d output bands, x%d"),
              sr_variant_name(model.variant), nin, nout, S);

    /* Every band the model reads is required; others are refused rather
     * than silently dropped. */
    for (b = 0; b < S2_NBANDS; b++) {
        int used = 0;

        for (i = 0; i < nin; i++)
            used |= model.in_bands[i] == (enum s2_band)b;
        if (used && !band_opt[b]->answer)
            G_fatal_error(_("The %s model needs option <%s> (%s)"),
                          sr_variant_name(model.variant), band_opt[b]->key,
                          band_suffix[b]);
        if (!used && band_opt[b]->answer)
            G_fatal_error(_("Option <%s> is not used by the %s model; "
                            "remove it or use a model that reads %s"),
                          band_opt[b]->key, sr_variant_name(model.variant),
                          band_suffix[b]);
    }
    for (i = 0; i < nin; i++)
        in_names[i] = band_opt[model.in_bands[i]]->answer;

    for (i = 0; i < nout; i++) {
        snprintf(out_names[i], GNAME_MAX, "%s_%s", output->answer,
                 band_suffix[model.out_bands[i]]);
        if (G_legal_filename(out_names[i]) < 0)
            G_fatal_error(_("<%s> is an illegal file name"), out_names[i]);
        if (G_find_raster2(out_names[i], G_mapset()) &&
            !G_check_overwrite(argc, argv))
            G_fatal_error(_("Raster map <%s> already exists"), out_names[i]);
    }

    G_get_window(&win);
    check_grid(&win, in_names, model.in_bands, nin);
    H = win.rows;
    W = win.cols;

    /* Read on the current region, write on the same extent with S times
     * finer cells. Only this process sees the output window. */
    Rast_set_input_window(&win);
    hr = win;
    hr.rows = H * S;
    hr.cols = W * S;
    G_adjust_Cell_head(&hr, 1, 1);
    Rast_set_output_window(&hr);

    for (i = 0; i < nin; i++)
        in_fd[i] = Rast_open_old(in_names[i], "");
    for (i = 0; i < nout; i++)
        out_fd[i] = Rast_open_fp_new(out_names[i]);

    /* Pad to at least one tile, and to an even size so that tiles start
     * on the 20 m grid; the padding repeats the last row and column. */
    Hp = H < T ? T : H + (H & 1);
    Wp = W < T ? T : W + (W & 1);
    nty = tile_layout(Hp, overlap, &ys, &yc);
    ntx = tile_layout(Wp, overlap, &xs, &xc);
    max_core = 0;
    for (ty = 0; ty < nty; ty++)
        if (yc[ty + 1] - yc[ty] > max_core)
            max_core = yc[ty + 1] - yc[ty];
    G_verbose_message(_("%d x %d tiles of %d x %d cells"), ntx, nty, T, T);

    tile_plane = (size_t)T * T;
    out_plane = tile_plane * S * S;
    hr_cols = (size_t)W * S;
    row_buf = Rast_allocate_f_input_buf();
    out_row = G_malloc(sizeof(FCELL) * hr_cols);
    strip = G_malloc(sizeof(float) * nin * T * Wp);
    null_strip = G_malloc((size_t)T * Wp);
    tile = G_malloc(sizeof(float) * nin * tile_plane);
    tile_out = G_malloc(sizeof(float) * nout * out_plane);
    hr_strip = G_malloc(sizeof(float) * nout * max_core * S * hr_cols);

    for (ty = 0; ty < nty; ty++) {
        int y0 = ys[ty], c0 = yc[ty], c1 = yc[ty + 1];

        G_percent(ty, nty, 1);
        if (c1 > H)
            c1 = H;

        /* Read the T rows of this tile row, as reflectances with nulls
         * set to 0 and flagged. */
        memset(null_strip, 0, (size_t)T * Wp);
        for (i = 0; i < nin; i++)
            for (r = 0; r < T; r++) {
                int src = y0 + r < H ? y0 + r : H - 1;
                float *dst = strip + ((size_t)i * T + r) * Wp;

                Rast_get_f_row(in_fd[i], row_buf, src);
                for (c = 0; c < Wp; c++) {
                    int sc = c < W ? c : W - 1;

                    if (Rast_is_f_null_value(&row_buf[sc])) {
                        dst[c] = 0.0f;
                        null_strip[(size_t)r * Wp + c] = 1;
                    }
                    else
                        dst[c] = (row_buf[sc] + offset) / scale;
                }
            }

        for (tx = 0; tx < ntx; tx++) {
            int x0 = xs[tx], d0 = xc[tx], d1 = xc[tx + 1];

            if (d1 > W)
                d1 = W;
            for (i = 0; i < nin; i++)
                for (r = 0; r < T; r++)
                    memcpy(tile + (i * T + r) * T,
                           strip + ((size_t)i * T + r) * Wp + x0,
                           sizeof(float) * T);

            sr_model_run(&model, tile, tile_out);

            /* Keep the core of the tile. */
            for (i = 0; i < nout; i++)
                for (r = (c0 - y0) * S; r < (c1 - y0) * S; r++) {
                    const float *src = tile_out + i * out_plane +
                                       (size_t)r * T * S + (d0 - x0) * S;
                    float *dst =
                        hr_strip +
                        ((size_t)i * max_core * S + r - (c0 - y0) * S) *
                            hr_cols +
                        (size_t)d0 * S;

                    memcpy(dst, src, sizeof(float) * (d1 - d0) * S);
                }
        }

        for (r = (c0 - y0) * S; r < (c1 - y0) * S; r++) {
            const char *nulls = null_strip + (size_t)(r / S) * Wp;

            for (i = 0; i < nout; i++) {
                const float *src =
                    hr_strip +
                    ((size_t)i * max_core * S + r - (c0 - y0) * S) * hr_cols;

                for (c = 0; c < (int)hr_cols; c++) {
                    if (nulls[c / S])
                        Rast_set_f_null_value(&out_row[c], 1);
                    else
                        out_row[c] = src[c] * scale - offset;
                }
                Rast_put_f_row(out_fd[i], out_row);
            }
        }
    }
    G_percent(1, 1, 1);

    for (i = 0; i < nin; i++)
        Rast_close(in_fd[i]);
    for (i = 0; i < nout; i++) {
        struct History hist;
        struct Colors colors;
        enum s2_band band = model.out_bands[i];
        char title[256];

        Rast_close(out_fd[i]);
        snprintf(title, sizeof(title), "Sentinel-2 %s super-resolved by %s",
                 band_suffix[band], sr_variant_name(model.variant));
        Rast_put_cell_title(out_names[i], title);
        if (Rast_read_colors(band_opt[band]->answer, "", &colors) > 0) {
            Rast_write_colors(out_names[i], G_mapset(), &colors);
            Rast_free_colors(&colors);
        }
        Rast_short_history(out_names[i], "raster", &hist);
        Rast_command_history(&hist);
        Rast_write_history(out_names[i], &hist);
    }

    G_free(row_buf);
    G_free(out_row);
    G_free(strip);
    G_free(null_strip);
    G_free(tile);
    G_free(tile_out);
    G_free(hr_strip);
    G_free(ys);
    G_free(yc);
    G_free(xs);
    G_free(xc);
    sr_model_free(&model);
    ocl_free(&ocl);

    G_message(_("Written %d raster maps at %g x %g map units"), nout, hr.ew_res,
              hr.ns_res);
    exit(EXIT_SUCCESS);
}
