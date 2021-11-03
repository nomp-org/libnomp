#if !defined(_LIB_GNOMP_IMPL_H_)
#define _LIB_GNOMP_IMPL_H_

#include <gnomp.h>

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GNOMP_OCL 0
#define GNOMP_CUDA 1
#define GNOMP_HIP 2

struct backend {
  int backend;
  void *bptr;
};

struct prog {
  int backend;
  void *bptr;
};

struct mem {
  size_t idx0, idx1;
  size_t usize;
  void *hptr;
  void *bptr;
};

union gnomp_arg {
  short s;
  unsigned short us;
  int i;
  unsigned int ui;
  long l;
  unsigned long ul;
  float f;
  double d;
  void *p;
};

int opencl_init(struct backend *ocl, const int platform_id,
                const int device_id);

int opencl_alloc(struct backend *ocl, struct mem *m);
int opencl_map(struct backend *ocl, struct mem *m, const int direction);

int opencl_get_mem_ptr(union gnomp_arg *arg, size_t *size, struct mem *m);

int opencl_build_knl(struct backend *ocl, struct prog *prg, const char *source,
                     const char *name);

int opencl_set_knl_arg(struct prog *prg, const int index, const size_t size,
                       void *arg);

int opencl_run_knl(struct backend *ocl, struct prog *prg, const int ndim,
                   const size_t *global, const size_t *local);

int opencl_finalize(struct backend *ocl);

#endif // _LIB_GNOMP_IMPL_H_
