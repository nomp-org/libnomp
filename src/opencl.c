#include "nomp-impl.h"

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// TODO: Handle errors properly in OpenCL backend
struct opencl_backend {
  cl_device_id device_id;
  cl_command_queue queue;
  cl_context ctx;
};

struct opencl_prog {
  cl_program prg;
  cl_kernel knl;
};

static int opencl_update(struct backend *bnd, struct mem *m, const int op) {
  struct opencl_backend *ocl = (struct opencl_backend *)bnd->bptr;

  cl_int err;
  if (op & NOMP_ALLOC) {
    m->bptr = tcalloc(cl_mem, 1);
    *((cl_mem *)m->bptr) =
        clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE,
                       (m->idx1 - m->idx0) * m->usize, NULL, &err);
    if (err != CL_SUCCESS) {
      tfree(m->bptr), m->bptr = NULL;
      return set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,
                     "create buffer", err);
    }
  }

  cl_mem *clm = (cl_mem *)m->bptr;
  if (op & NOMP_TO) {
    err = clEnqueueWriteBuffer(
        ocl->queue, *clm, CL_TRUE, 0, (m->idx1 - m->idx0) * m->usize,
        (char *)m->hptr + m->idx0 * m->usize, 0, NULL, NULL);
  } else if (op == NOMP_FROM) {
    err = clEnqueueReadBuffer(
        ocl->queue, *clm, CL_TRUE, 0, (m->idx1 - m->idx0) * m->usize,
        (char *)m->hptr + m->idx0 * m->usize, 0, NULL, NULL);
  } else if (op == NOMP_FREE) {
    err = clReleaseMemObject(*clm);
    if (err == CL_SUCCESS)
      tfree(m->bptr), m->bptr = NULL;
  }

  // FIXME: Wrong. call set_log()
  return err != CL_SUCCESS;
}

static int opencl_knl_build(struct backend *bnd, struct prog *prg,
                            const char *source, const char *name) {
  struct opencl_backend *ocl = bnd->bptr;
  struct opencl_prog *ocl_prg = prg->bptr = tcalloc(struct opencl_prog, 1);

  cl_int err;
  ocl_prg->prg = clCreateProgramWithSource(
      ocl->ctx, 1, (const char **)(&source), NULL, &err);
  if (err != CL_SUCCESS)
    return set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,
                   "kernel build", err);

  err = clBuildProgram(ocl_prg->prg, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    // Determine log size
    size_t log_size;
    clGetProgramBuildInfo(ocl_prg->prg, ocl->device_id, CL_PROGRAM_BUILD_LOG, 0,
                          NULL, &log_size);

    // Allocate memory for the log
    char *log = tcalloc(char, log_size);
    // Verify log memory allocation
    if (!log)
      return set_log(NOMP_RUNTIME_MEMORY_ALLOCATION_FAILED, NOMP_ERROR,
                     ERR_STR_RUNTIME_MEMORY_ALLOCATION_FAILURE);

    // Get the log
    clGetProgramBuildInfo(ocl_prg->prg, ocl->device_id, CL_PROGRAM_BUILD_LOG,
                          log_size, log, NULL);
    // Print the log
    printf("clBuildProgram error: %s\n", log);

    ocl_prg->prg = NULL;
    ocl_prg->knl = NULL;

    return set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,
                   "kernel build", err);
  }

  ocl_prg->knl = clCreateKernel(ocl_prg->prg, name, &err);
  if (err != CL_SUCCESS) {
    ocl_prg->knl = NULL;
    return set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,
                   "kernel create", err);
  }

  return 0;
}

static int opencl_knl_run(struct backend *bnd, struct prog *prg) {
  struct opencl_prog *oprg = (struct opencl_prog *)prg->bptr;

  for (int i = 0; i < prg->narg; i++) {
    void *p = prg->args[i].ptr;
    size_t size = prg->args[i].size;
    unsigned type = prg->args[i].type;
    if (type == NOMP_PTR) {
      size = sizeof(cl_mem);
      p = (void *)(*((cl_mem **)p));
    }

    cl_int err = clSetKernelArg(oprg->knl, i, size, p);
    if (err != CL_SUCCESS) {
      return set_log(
          NOMP_OPENCL_FAILURE, NOMP_ERROR,
          "OpenCL clSetKernelArg() failed for argument %d (ptr = %p).", i, p);
    }
  }

  // FIXME: May be do this differently?
  size_t global[3];
  for (unsigned i = 0; i < prg->ndim; i++)
    global[i] = prg->global[i] * prg->local[i];

  struct opencl_backend *ocl = (struct opencl_backend *)bnd->bptr;
  cl_int err = clEnqueueNDRangeKernel(ocl->queue, oprg->knl, prg->ndim, NULL,
                                      global, prg->local, 0, NULL, NULL);
  // FIXME: Wrong. Call set_log()
  return err != CL_SUCCESS;
}

static int opencl_knl_free(struct prog *prg) {
  struct opencl_prog *oprg = prg->bptr;

  cl_int err = clReleaseKernel(oprg->knl);
  if (err != CL_SUCCESS) {
    return set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,
                   "kernel release", err);
  }

  err = clReleaseProgram(oprg->prg);
  if (err != CL_SUCCESS) {
    return set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,
                   "program release", err);
  }

  tfree(prg->bptr), prg->bptr = NULL;

  return 0;
}

static int opencl_finalize(struct backend *bnd) {
  struct opencl_backend *ocl = bnd->bptr;

  cl_int err = clReleaseCommandQueue(ocl->queue);
  if (err != CL_SUCCESS) {
    return set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,
                   "command queue release", err);
  }

  err = clReleaseContext(ocl->ctx);
  if (err != CL_SUCCESS) {
    return set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,
                   "context release", err);
  }

  tfree(bnd->bptr), bnd->bptr = NULL;

  return 0;
}

int opencl_init(struct backend *bnd, const int platform_id,
                const int device_id) {
  cl_uint num_platforms;
  cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
  // TODO: Check for the retunr value of `err` as well.
  if (platform_id < 0 | platform_id >= num_platforms) {
    return set_log(NOMP_USER_PLATFORM_IS_INVALID, NOMP_ERROR,
                   "Platform id %d provided to libnomp is not valid.",
                   platform_id);
  }

  cl_platform_id *cl_platforms = tcalloc(cl_platform_id, num_platforms);
  if (cl_platforms == NULL) {
    return set_log(NOMP_RUNTIME_MEMORY_ALLOCATION_FAILED, NOMP_ERROR,
                   ERR_STR_RUNTIME_MEMORY_ALLOCATION_FAILURE);
  }

  err = clGetPlatformIDs(num_platforms, cl_platforms, &num_platforms);
  cl_platform_id platform = cl_platforms[platform_id];
  tfree(cl_platforms);

  cl_uint num_devices;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  if (device_id < 0 || device_id >= num_devices) {
    return set_log(NOMP_USER_DEVICE_IS_INVALID, NOMP_ERROR,
                   ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }

  cl_device_id *cl_devices = tcalloc(cl_device_id, num_devices);
  if (cl_devices == NULL) {
    return set_log(NOMP_RUNTIME_MEMORY_ALLOCATION_FAILED, NOMP_ERROR,
                   ERR_STR_RUNTIME_MEMORY_ALLOCATION_FAILURE);
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, cl_devices,
                       &num_devices);
  cl_device_id device = cl_devices[device_id];
  tfree(cl_devices);

  struct opencl_backend *ocl = tcalloc(struct opencl_backend, 1);
  ocl->device_id = device;
  ocl->ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  ocl->queue = clCreateCommandQueueWithProperties(ocl->ctx, device, 0, &err);

  bnd->bptr = (void *)ocl;
  bnd->update = opencl_update;
  bnd->knl_build = opencl_knl_build;
  bnd->knl_run = opencl_knl_run;
  bnd->knl_free = opencl_knl_free;
  bnd->finalize = opencl_finalize;

  return 0;
}
