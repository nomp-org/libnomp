#include <gnomp-impl.h>

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

struct opencl_backend {
  cl_command_queue queue;
  cl_context ctx;
};

int opencl_init(struct backend *bnd, const int platform_id,
                const int device_id) {
  cl_uint num_platforms;
  cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
  // TODO: check err
  if (platform_id < 0 | platform_id >= num_platforms)
    return GNOMP_INVALID_PLATFORM;

  cl_platform_id *cl_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  if (cl_platforms == NULL)
    return GNOMP_MALLOC_ERROR;

  err = clGetPlatformIDs(num_platforms, cl_platforms, &num_platforms);
  cl_platform_id platform = cl_platforms[platform_id];

  cl_uint num_devices;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  if (device_id < 0 || device_id >= num_devices)
    return GNOMP_INVALID_DEVICE;

  cl_device_id *cl_devices = calloc(num_devices, sizeof(cl_device_id));
  if (cl_devices == NULL)
    return GNOMP_MALLOC_ERROR;

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, cl_devices,
                       &num_devices);
  cl_device_id device = cl_devices[device_id];

  bnd->backend = GNOMP_OCL;
  struct opencl_backend *ocl = bnd->bptr =
      calloc(1, sizeof(struct opencl_backend));
  ocl->ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  ocl->queue = clCreateCommandQueueWithProperties(ocl->ctx, device, NULL, &err);

  free(cl_devices);
  free(cl_platforms);

  return 0;
}

struct opencl_mem {
  cl_mem mem;
};

int opencl_map(struct backend *bnd, struct mem *m, const int direction,
               const int alloc) {
  struct opencl_backend *ocl = bnd->bptr;
  cl_int err;
  if (alloc) {
    struct opencl_mem *ocl_mem = m->bptr = calloc(1, sizeof(struct opencl_mem));
    ocl_mem->mem = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE,
                                  (m->idx1 - m->idx0) * m->usize, NULL, &err);
    if (err != CL_SUCCESS)
      return 1;
  }

  struct opencl_mem *ocl_mem = m->bptr;
  if (direction == GNOMP_H2D)
    err = clEnqueueWriteBuffer(ocl->queue, ocl_mem->mem, CL_TRUE, 0,
                               (m->idx1 - m->idx0) * m->usize, m->hptr, 0, NULL,
                               NULL);
  else if (direction == GNOMP_D2H)
    err = clEnqueueReadBuffer(ocl->queue, ocl_mem->mem, CL_TRUE, 0,
                              (m->idx1 - m->idx0) * m->usize, m->hptr, 0, NULL,
                              NULL);

  if (err != CL_SUCCESS)
    return 1;

  return 0;
}

int opencl_get_mem_ptr(union gnomp_arg *arg, size_t *size, struct mem *m) {
  struct opencl_mem *mem = m->bptr;
  arg->p = mem->mem;
  *size = sizeof(cl_mem);
  return 0;
}

struct opencl_prog {
  cl_program prg;
  cl_kernel knl;
};

int opencl_build_knl(struct backend *bnd, struct prog *prg, const char *source,
                     const char *name) {
  struct opencl_backend *ocl = bnd->bptr;
  struct opencl_prog *ocl_prg = prg->bptr =
      calloc(1, sizeof(struct opencl_prog));

  cl_int err;
  ocl_prg->prg = clCreateProgramWithSource(
      ocl->ctx, 1, (const char **)(&source), NULL, &err);
  if (err != CL_SUCCESS)
    return 1;

  err = clBuildProgram(ocl_prg->prg, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    ocl_prg->prg = NULL;
    ocl_prg->knl = NULL;
    return 1;
  }

  ocl_prg->knl = clCreateKernel(ocl_prg->prg, name, &err);
  if (err != CL_SUCCESS) {
    ocl_prg->knl = NULL;
    return 1;
  }

  return 0;
}

int opencl_set_knl_arg(struct prog *prg, const int index, const size_t size,
                       void *arg) {
  struct opencl_prog *ocl_prg = prg->bptr;
  cl_int err = clSetKernelArg(ocl_prg->knl, index, size, arg);
  if (err != CL_SUCCESS)
    return 1;
  return 0;
}

int opencl_run_knl(struct backend *bnd, struct prog *prg, const int ndim,
                   const size_t *global, const size_t *local) {
  struct opencl_backend *ocl = bnd->bptr;
  struct opencl_prog *ocl_prg = prg->bptr;
  cl_int err = clEnqueueNDRangeKernel(ocl->queue, ocl_prg->knl, ndim, NULL,
                                      global, local, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    return 1;
  return 0;
}

int opencl_finalize(struct backend *bnd) {
  struct opencl_backend *ocl = bnd->bptr;
  if (ocl != NULL) {
    cl_int err = clReleaseCommandQueue(ocl->queue);
    if (err != CL_SUCCESS)
      return 1;
    err = clReleaseContext(ocl->ctx);
    if (err != CL_SUCCESS)
      return 1;
    free(ocl);
  }

  return 0;
}

#undef set_knl_arg
