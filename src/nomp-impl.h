#if !defined(_LIB_NOMP_IMPL_H_)
#define _LIB_NOMP_IMPL_H_

#include <nomp.h>

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NOMP_OCL 0
#define NOMP_CUDA 1
#define NOMP_HIP 2

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

int opencl_init(struct backend *ocl, const int platform_id,
                const int device_id);

int opencl_map(struct backend *ocl, struct mem *m, const int direction);
void opencl_map_ptr(void **p, size_t *size, struct mem *m);

int opencl_knl_build(struct backend *ocl, struct prog *prg, const char *source,
                     const char *name);
int opencl_knl_set(struct prog *prg, const int index, const size_t size,
                   void *arg);
int opencl_knl_run(struct backend *ocl, struct prog *prg, const int ndim,
                   const size_t *global, const size_t *local);
int opencl_knl_free(struct prog *prg);

int opencl_finalize(struct backend *ocl);

#endif // _LIB_NOMP_IMPL_H_
