#include "nomp-impl.h"

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

static int opencl_map(struct backend *bnd, struct mem *m, const int op);
static void opencl_map_ptr(void **p, size_t *size, struct mem *m);
static int opencl_knl_build(struct backend *bnd, struct prog *prg,
                            const char *source, const char *name);
static int opencl_knl_set(struct prog *prg, const int index, const size_t size,
                          void *arg);
static int opencl_knl_run(struct backend *bnd, struct prog *prg, const int ndim,
                          const size_t *global, const size_t *local);
static int opencl_knl_free(struct prog *prg);
static int opencl_finalize(struct backend *bnd);

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
    return NOMP_INVALID_PLATFORM;

  cl_platform_id *cl_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  if (cl_platforms == NULL)
    return NOMP_MALLOC_ERROR;

  err = clGetPlatformIDs(num_platforms, cl_platforms, &num_platforms);
  cl_platform_id platform = cl_platforms[platform_id];

  cl_uint num_devices;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  if (device_id < 0 || device_id >= num_devices)
    return NOMP_INVALID_DEVICE;

  cl_device_id *cl_devices = calloc(num_devices, sizeof(cl_device_id));
  if (cl_devices == NULL)
    return NOMP_MALLOC_ERROR;

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, cl_devices,
                       &num_devices);
  cl_device_id device = cl_devices[device_id];

  struct opencl_backend *ocl = bnd->bptr =
      calloc(1, sizeof(struct opencl_backend));
  ocl->ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  ocl->queue = clCreateCommandQueueWithProperties(ocl->ctx, device, NULL, &err);

  free(cl_devices);
  free(cl_platforms);

  bnd->map = opencl_map;
  bnd->map_ptr = opencl_map_ptr;
  bnd->knl_build = opencl_knl_build;
  bnd->knl_set = opencl_knl_set;
  bnd->knl_run = opencl_knl_run;
  bnd->knl_free = opencl_knl_free;
  bnd->finalize = opencl_finalize;

  return 0;
}

struct opencl_mem {
  cl_mem mem;
};

static int opencl_map(struct backend *bnd, struct mem *m, const int op) {
  struct opencl_backend *ocl = bnd->bptr;

  cl_int err;
  if (op & NOMP_ALLOC) {
    struct opencl_mem *ocl_mem = m->bptr = calloc(1, sizeof(struct opencl_mem));
    ocl_mem->mem = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE,
                                  (m->idx1 - m->idx0) * m->usize, NULL, &err);
    if (err != CL_SUCCESS)
      return 1;
  }

  if (op & NOMP_H2D) {
    struct opencl_mem *ocl_mem = m->bptr;
    err = clEnqueueWriteBuffer(
        ocl->queue, ocl_mem->mem, CL_TRUE, m->idx0 * m->usize,
        (m->idx1 - m->idx0) * m->usize, m->hptr, 0, NULL, NULL);
    return err != CL_SUCCESS;
  }

  struct opencl_mem *ocl_mem = m->bptr;
  if (op == NOMP_D2H) {
    err = clEnqueueReadBuffer(ocl->queue, ocl_mem->mem, CL_TRUE, 0,
                              (m->idx1 - m->idx0) * m->usize, m->hptr, 0, NULL,
                              NULL);

    return err != CL_SUCCESS;
  } else if (op == NOMP_FREE) {
    err = clReleaseMemObject(ocl_mem->mem);
    if (err != CL_SUCCESS)
      return 1;
    free(m->bptr), m->bptr = NULL;
  }

  return 0;
}

static void opencl_map_ptr(void **p, size_t *size, struct mem *m) {
  struct opencl_mem *ocl_mem = m->bptr;
  *p = (void *)&ocl_mem->mem;
  *size = sizeof(cl_mem);
}

struct opencl_prog {
  cl_program prg;
  cl_kernel knl;
};

static int opencl_knl_build(struct backend *bnd, struct prog *prg,
                            const char *source, const char *name) {
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

static int opencl_knl_set(struct prog *prg, const int index, const size_t size,
                          void *arg) {
  struct opencl_prog *ocl_prg = prg->bptr;
  cl_int err = clSetKernelArg(ocl_prg->knl, index, size, arg);
  return err != CL_SUCCESS;
}

static int opencl_knl_run(struct backend *bnd, struct prog *prg, const int ndim,
                          const size_t *global, const size_t *local) {
  struct opencl_backend *ocl = bnd->bptr;
  struct opencl_prog *ocl_prg = prg->bptr;
  cl_int err = clEnqueueNDRangeKernel(ocl->queue, ocl_prg->knl, ndim, NULL,
                                      global, local, 0, NULL, NULL);
  return err != CL_SUCCESS;
}

static int opencl_knl_free(struct prog *prg) {
  struct opencl_prog *ocl_prg = prg->bptr;
  cl_int err = clReleaseKernel(ocl_prg->knl);
  if (err != CL_SUCCESS)
    return 1;
  err = clReleaseProgram(ocl_prg->prg);
  if (err != CL_SUCCESS)
    return 1;
  free(prg->bptr), prg->bptr = NULL;

  return 0;
}

static int opencl_finalize(struct backend *bnd) {
  struct opencl_backend *ocl = bnd->bptr;
  cl_int err = clReleaseCommandQueue(ocl->queue);
  if (err != CL_SUCCESS)
    return 1;
  err = clReleaseContext(ocl->ctx);
  if (err != CL_SUCCESS)
    return 1;
  free(bnd->bptr), bnd->bptr = NULL;

  return 0;
}
