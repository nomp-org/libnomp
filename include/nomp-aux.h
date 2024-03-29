#if !defined(_NOMP_AUX_H_)
#define _NOMP_AUX_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_other_utils Other helper functions.
 *
 * @brief Various helper functions used internally by libnomp core library.
 */

char *nomp_str_cat(unsigned n, unsigned max_len, ...);

int nomp_str_toui(const char *str, size_t size);

int nomp_max(unsigned n, ...);

char *nomp_copy_env(const char *name, size_t size);

int nomp_path_len(size_t *len, const char *path);

#ifdef __cplusplus
}
#endif

#endif // _NOMP_AUX_H_
