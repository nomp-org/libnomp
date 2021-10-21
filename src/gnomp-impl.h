#if !defined(_LIB_GNOMP_IMPL_H_)
#define _LIB_GNOMP_IMPL_H_

#include <gnomp.h>

#define GNOMP_OCL 1
#define GNOMP_CUDA 2
#define GNOMP_HIP 4

struct backend {
  int backend;
  cl_command_queue queue;
  cl_context context;
};

struct mem {
  cl_mem d_ptr;
  void *h_ptr;
  size_t size, usize;
};

int opencl_init(struct backend *ocl, int platform_id, int device_id);
int opencl_map_to(struct backend *ocl, struct mem *m, void *ptr, size_t id0,
                  size_t id1, size_t usize, int alloc);
int opencl_map_from(struct backend *ocl, struct mem *m, size_t id0, size_t id1,
                    size_t usize);

#endif // _LIB_GNOMP_IMPL_H_
