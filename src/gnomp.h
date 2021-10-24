#if !defined(_LIB_GNOMP_H_)
#define _LIB_GNOMP_H_

#define CL_TARGET_OPENCL_VERSION 220

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GNOMP_INVALID_BACKEND -1
#define GNOMP_INVALID_PLATFORM -2
#define GNOMP_INVALID_DEVICE -4

#define GNOMP_MALLOC_ERROR -128

#define GNOMP_INVALID_MAP_PTR -256

#ifdef __cplusplus
extern "C" {
#endif

int gnomp_init(char *backend, int platform, int device);

int gnomp_map(void *ptr, size_t start_idx, size_t end_idx, size_t unit_size,
              int direction, int handle);

int gnomp_run(int *id, const char *source, const char *name, int handle,
              int nargs, ...);

#ifdef __cplusplus
}
#endif

#endif // _LIB_GNOMP_H_
