#if !defined(_LIB_GNOMP_IMPL_H_)
#define _LIB_GNOMP_IMPL_H_

#include <gnomp.h>

#define GNOMP_OCL 1
#define GNOMP_CUDA 2
#define GNOMP_HIP 4

#define GNOMP_H2D 0
#define GNOMP_D2H 1

struct backend {
  int backend;
  cl_command_queue queue;
  cl_context ctx;
};

struct mem {
  cl_mem dptr;
  void *hptr;
  size_t size, usize;
};

struct prog {
  cl_program prg;
  cl_kernel knl;
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

int opencl_map(struct backend *ocl, struct mem *m, void *ptr, const size_t id0,
               const size_t id1, const size_t usize, const int direction,
               const int alloc);

int opencl_build_knl(struct backend *ocl, struct prog *prg, const char *source,
                     const char *name);

int opencl_set_knl_arg(struct prog *prg, const int index, const size_t size,
                       void *arg);

int opencl_run_knl(struct backend *ocl, struct prog *prg, const int ndim,
                   const size_t *global, const size_t *local);

#endif // _LIB_GNOMP_IMPL_H_
