#ifndef I_SEN2SR_OPENCL_SEN2SR_H
#define I_SEN2SR_OPENCL_SEN2SR_H

#include "ocl_backend.h"
#include "sen2sr_full.h"
#include "sen2sr_model.h"

/* Dispatcher for the bundled SEN2SRLite models and the imported full
 * SEN2SR model. The GRASS module only talks to this interface. */
enum sen2sr_engine { SEN2SR_LITE, SEN2SR_FULL };

struct sen2sr {
    enum sen2sr_engine engine;
    int scale, nin, nout;
    enum s2_band in_bands[S2_NBANDS], out_bands[S2_NBANDS];
    struct sr_model lite;
    struct sr_full full;
};

/* Find a model directory below a search root such as $HOME/models/. The
 * known published layouts are tried in order. Returns 0 if not found. */
int sen2sr_default_model(const char *root, const char *variant, char *path,
                         size_t size);

/* As above, but download the published weights into <root> when the
 * requested variant is missing there. Only used for the default root. */
int sen2sr_ensure_default_model(const char *root, const char *variant,
                                char *path, size_t size);

/* variant is the module's model option: auto, main, rgbn_x4, rswir_x2
 * or full. With auto, sr_model_load() detects the Lite variant; full is
 * only loaded when explicitly requested. */
void sen2sr_load(struct sen2sr *s, struct ocl_backend *ocl, const char *dir,
                 const char *variant);
void sen2sr_run(struct sen2sr *s, const float *in, float *out);
void sen2sr_free(struct sen2sr *s);
const char *sen2sr_name(const struct sen2sr *s);

#endif /* I_SEN2SR_OPENCL_SEN2SR_H */
