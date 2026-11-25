#ifndef I_SEN2SR_OPENCL_UTIL_H
#define I_SEN2SR_OPENCL_UTIL_H

/* Compatibility shim for the imported full-SEN2SR sources. They were
 * written against tiny stand-ins for the GRASS message helpers; here the
 * real helpers are available, so map the two names they use onto them. */
#include <grass/gis.h>
#include <grass/glocale.h>

#define die G_fatal_error
#define msg G_message

#endif /* I_SEN2SR_OPENCL_UTIL_H */
