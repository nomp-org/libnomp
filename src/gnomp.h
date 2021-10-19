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

#define GNOMP_INVALID_BACKEND 1
#define GNOMP_INVALID_PLATFORM 2
#define GNOMP_INVALID_DEVICE 4

#define GNOMP_MALLOC_ERROR 1024

#ifdef __cplusplus
extern "C" {
#endif

int gnomp_init(char *backend, int platform, int device);

#ifdef __cplusplus
}
#endif

#endif // _LIB_GNOMP_H_
