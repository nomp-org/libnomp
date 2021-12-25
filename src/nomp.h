#if !defined(_LIB_NOMP_H_)
#define _LIB_NOMP_H_

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* Map Direction */
#define NOMP_ALLOC 1
#define NOMP_H2D 2
#define NOMP_D2H 4
#define NOMP_FREE 8

/* Errors */
#define NOMP_INVALID_BACKEND -1
#define NOMP_INVALID_PLATFORM -2
#define NOMP_INVALID_DEVICE -4
#define NOMP_INVALID_TYPE -8
#define NOMP_INVALID_MAP_PTR -16
#define NOMP_MALLOC_ERROR -32
#define NOMP_INVALID_MAP_OP -64
#define NOMP_INITIALIZED_ERROR -128
#define NOMP_NOT_INITIALIZED_ERROR -256
#define NOMP_KNL_BUILD_ERROR -512
#define NOMP_KNL_ARG_TYPE_ERROR -1024
#define NOMP_KNL_ARG_SET_ERROR -2048
#define NOMP_KNL_RUN_ERROR -4096
#define NOMP_INVALID_KNL -8192
#define NOMP_FINALIZE_FAILURE -16384

/* Types */
#define NOMP_SCALAR 1
#define NOMP_PTR 2

/* Functions */
#ifdef __cplusplus
extern "C" {
#endif

int nomp_init(const char *backend, const int platform, const int device);

int nomp_map(void *ptr, const size_t start_idx, const size_t end_idx,
             const size_t unit_size, const int op);

int nomp_run(int *id, const char *source, const char *name, const int ndim,
             const size_t *global, const size_t *local, const int nargs, ...);

int nomp_err_str(int err_id, char *buf, int buf_size);

int nomp_finalize(void);

#define nomp_check_err_(err, file, line)                                       \
  {                                                                            \
    if (err != 0) {                                                            \
      char buf[2 * BUFSIZ];                                                    \
      nomp_err_str(err, buf, 2 * BUFSIZ);                                      \
      printf("%s:%d %s\n", file, line, buf);                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define nomp_check_err(err) nomp_check_err_(err, __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_H_
