#if !defined(_LIB_GNOMP_H_)
#define _LIB_GNOMP_H_

#include <stddef.h>

/* Map Direction */
#define GNOMP_H2D 0
#define GNOMP_D2H 1

/* Errors */
#define GNOMP_INVALID_BACKEND -1
#define GNOMP_INVALID_PLATFORM -2
#define GNOMP_INVALID_DEVICE -4
#define GNOMP_INVALID_TYPE -8
#define GNOMP_INVALID_MAP_PTR -16

#define GNOMP_MALLOC_ERROR -128

/* Types */
#define GNOMP_SHORT 0
#define GNOMP_USHORT 1
#define GNOMP_INT 10
#define GNOMP_UINT 11
#define GNOMP_LONG 20
#define GNOMP_ULONG 21
#define GNOMP_FLOAT 30
#define GNOMP_DOUBLE 31
#define GNOMP_PTR 50

/* Functions */
#ifdef __cplusplus
extern "C" {
#endif

int gnomp_init(int *handle, const char *backend, const int platform,
               const int device);

int gnomp_map(void *ptr, const size_t start_idx, const size_t end_idx,
              const size_t unit_size, const int direction, const int handle);

int gnomp_run(int *id, const char *source, const char *name, const int handle,
              const int ndim, const size_t *global, const size_t *local,
              const int nargs, ...);

#ifdef __cplusplus
}
#endif

#endif // _LIB_GNOMP_H_
