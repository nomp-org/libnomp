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
  cl_mem d_ptr;
  void *h_ptr;
  size_t size, usize;
};

struct prog {
  cl_program prg;
};

int opencl_init(struct backend *ocl, int platform_id, int device_id);

int opencl_map(struct backend *ocl, struct mem *m, void *ptr, size_t id0,
               size_t id1, size_t usize, int direction, int alloc);

int opencl_build_program(struct backend *ocl, struct prog *prg,
                         const char *source);

#endif // _LIB_GNOMP_IMPL_H_
