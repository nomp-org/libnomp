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

int gnomp_map_to(void *ptr, size_t id0, size_t id1, size_t usize, int handle);

int gnomp_map_from(void *ptr, size_t id0, size_t id1, size_t usize, int handle);

#ifdef __cplusplus
}
#endif

#endif // _LIB_GNOMP_H_
