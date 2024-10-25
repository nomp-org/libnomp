#include "nomp-impl.h"

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#include <OpenCL/cl.h>
#define clCreateCommandQueueWithProperties clCreateCommandQueue
#else
#include <CL/cl.h>
#endif

static const char *ERR_STR_OPENCL_FAILURE = "%s failed with error: %d (%s).";

#define CASE(MSG, VAL, STR)                                                    \
  case VAL:                                                                    \
    return nomp_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,   \
                    MSG, VAL, STR);                                            \
    break;

// clang-format off
#define FOR_EACH_ERROR(S)                                                      \
  CASE(S, CL_INVALID_BINARY, "CL_INVALID_BINARY")                              \
  CASE(S, CL_INVALID_BUFFER_SIZE, "CL_INVALID_BUFFER_SIZE")                    \
  CASE(S, CL_INVALID_COMMAND_QUEUE, "CL_INVALID_COMMAND_QUEUE")                \
  CASE(S, CL_INVALID_CONTEXT, "CL_INVALID_CONTEXT")                            \
  CASE(S, CL_INVALID_DEVICE, "CL_INVALID_DEVICE")                              \
  CASE(S, CL_INVALID_EVENT, "CL_INVALID_EVENT")                                \
  CASE(S, CL_INVALID_EVENT_WAIT_LIST, "CL_INVALID_EVENT_WAIT_LIST")            \
  CASE(S, CL_INVALID_GLOBAL_OFFSET, "CL_INVALID_GLOBAL_OFFSET")                \
  CASE(S, CL_INVALID_GLOBAL_WORK_SIZE, "CL_INVALID_GLOBAL_WORK_SIZE")          \
  CASE(S, CL_INVALID_KERNEL, "CL_INVALID_KERNEL")                              \
  CASE(S, CL_INVALID_KERNEL_ARGS, "CL_INVALID_KERNEL_ARGS")                    \
  CASE(S, CL_INVALID_KERNEL_DEFINITION, "CL_INVALID_KERNEL_DEFINITION")        \
  CASE(S, CL_INVALID_KERNEL_NAME, "CL_INVALID_KERNEL_NAME")                    \
  CASE(S, CL_INVALID_MEM_OBJECT, "CL_INVALID_MEM_OBJECT")                      \
  CASE(S, CL_INVALID_OPERATION, "CL_INVALID_OPERATION")                        \
  CASE(S, CL_INVALID_PROGRAM, "CL_INVALID_PROGRAM")                            \
  CASE(S, CL_INVALID_PROGRAM_EXECUTABLE, "CL_INVALID_PROGRAM_EXECUTABLE")      \
  CASE(S, CL_INVALID_VALUE, "CL_INVALID_VALUE")                                \
  CASE(S, CL_INVALID_WORK_DIMENSION, "CL_INVALID_WORK_DIMENSION")              \
  CASE(S, CL_INVALID_WORK_GROUP_SIZE, "CL_INVALID_WORK_GROUP_SIZE")            \
  CASE(S, CL_INVALID_WORK_ITEM_SIZE, "CL_INVALID_WORK_ITEM_SIZE")              \
  CASE(S, CL_MEM_OBJECT_ALLOCATION_FAILURE, "CL_MEM_OBJECT_ALLOCATION_FAILURE")\
  CASE(S, CL_MISALIGNED_SUB_BUFFER_OFFSET, "CL_MISALIGNED_SUB_BUFFER_OFFSET")  \
  CASE(S, CL_OUT_OF_RESOURCES, "CL_OUT_OF_RESOURCES")                          \
  CASE(S, CL_OUT_OF_HOST_MEMORY, "CL_OUT_OF_HOST_MEMORY")
// clang-format on

#define check(call, msg)                                                       \
  {                                                                            \
    cl_int err_ = (call);                                                      \
    if (err_ != CL_SUCCESS) {                                                  \
      switch (err_) {                                                          \
        FOR_EACH_ERROR(msg)                                                    \
      default:                                                                 \
        return nomp_log(NOMP_OPENCL_FAILURE, NOMP_ERROR,                       \
                        ERR_STR_OPENCL_FAILURE, msg, err_, "UNKNOWN");         \
      }                                                                        \
    }                                                                          \
  }

struct opencl_backend_t {
  cl_device_id     device_id;
  cl_command_queue queue;
  cl_context       ctx;
};

struct opencl_prog_t {
  cl_program prg;
  cl_kernel  knl;
};

static int opencl_update(nomp_backend_t *bnd, nomp_mem_t *m,
                         const nomp_map_direction_t op, size_t start,
                         size_t end, size_t usize) {
  struct opencl_backend_t *ocl = (struct opencl_backend_t *)bnd->bptr;

  cl_int err;
  if (op & NOMP_ALLOC) {
    cl_mem *clm = nomp_calloc(cl_mem, 1);
    *clm        = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE,
                                 NOMP_MEM_BYTES(start, end, usize), NULL, &err);
    check(err, "clCreateBuffer");
    m->bptr = (void *)clm, m->bsize = sizeof(cl_mem);
  }

  cl_mem *clm = (cl_mem *)m->bptr;
  if (op & NOMP_TO) {
    check(clEnqueueWriteBuffer(ocl->queue, *clm, CL_TRUE,
                               NOMP_MEM_OFFSET(start - m->idx0, usize),
                               NOMP_MEM_BYTES(start, end, usize),
                               (char *)m->hptr + NOMP_MEM_OFFSET(start, usize),
                               0, NULL, NULL),
          "clEnqueueWriteBuffer");
  } else if (op == NOMP_FROM) {
    check(clEnqueueReadBuffer(ocl->queue, *clm, CL_TRUE,
                              NOMP_MEM_OFFSET(start - m->idx0, usize),
                              NOMP_MEM_BYTES(start, end, usize),
                              (char *)m->hptr + NOMP_MEM_OFFSET(start, usize),
                              0, NULL, NULL),
          "clEnqueueReadBuffer");
  } else if (op == NOMP_FREE) {
    check(clReleaseMemObject(*clm), "clReleaseMemObject");
    nomp_free(&m->bptr);
  }

  return 0;
}

static int opencl_knl_build(nomp_backend_t *bnd, nomp_prog_t *prg,
                            const char *source, const char *name) {
  struct opencl_prog_t *ocl_prg = nomp_calloc(struct opencl_prog_t, 1);

  struct opencl_backend_t *ocl = (struct opencl_backend_t *)bnd->bptr;
  cl_int                   err;
  ocl_prg->prg = clCreateProgramWithSource(
      ocl->ctx, 1, (const char **)(&source), NULL, &err);
  check(err, "clCreateProgramWithSource");

  err = clBuildProgram(ocl_prg->prg, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    ocl_prg->prg = NULL, ocl_prg->knl = NULL;

    size_t log_size;
    clGetProgramBuildInfo(ocl_prg->prg, ocl->device_id, CL_PROGRAM_BUILD_LOG, 0,
                          NULL, &log_size);
    char *log = nomp_calloc(char, log_size);
    clGetProgramBuildInfo(ocl_prg->prg, ocl->device_id, CL_PROGRAM_BUILD_LOG,
                          log_size, log, NULL);
    int err = nomp_log(NOMP_OPENCL_FAILURE, NOMP_ERROR,
                       "clBuildProgram failed with error:\n %s.", log);
    nomp_free(&log);
    return err;
  }

  ocl_prg->knl = clCreateKernel(ocl_prg->prg, name, &err);
  check(err, "clCreateKernel");
  prg->bptr = (void *)ocl_prg;

  return 0;
}

static int opencl_knl_run(nomp_backend_t *bnd, nomp_prog_t *prg) {
  struct opencl_prog_t *ocl_prg = (struct opencl_prog_t *)prg->bptr;

  for (unsigned i = 0; i < prg->nargs; i++) {
    check(clSetKernelArg(ocl_prg->knl, i, prg->args[i].size, prg->args[i].ptr),
          "clSetKernelArg");
  }

  struct opencl_backend_t *ocl = (struct opencl_backend_t *)bnd->bptr;
  check(clEnqueueNDRangeKernel(ocl->queue, ocl_prg->knl, prg->ndim, NULL,
                               prg->gws, prg->local, 0, NULL, NULL),
        "clEnqueueNDRangeKernel");
  check(clFinish(ocl->queue), "clFinish");

  return 0;
}

static int opencl_knl_free(nomp_prog_t *prg) {
  struct opencl_prog_t *ocl_prg = (struct opencl_prog_t *)prg->bptr;

  if (ocl_prg) {
    check(clReleaseKernel(ocl_prg->knl), "clReleaseKernel");
    check(clReleaseProgram(ocl_prg->prg), "clReleaseProgram");
  }

  nomp_free(&prg->bptr);
  return 0;
}

static int opencl_sync(nomp_backend_t *bnd) {
  struct opencl_backend_t *ocl = (struct opencl_backend_t *)bnd->bptr;
  check(clFinish(ocl->queue), "clFinish");
  return 0;
}

static int opencl_finalize(nomp_backend_t *bnd) {
  struct opencl_backend_t *ocl = (struct opencl_backend_t *)bnd->bptr;

  if (ocl) {
    check(clReleaseCommandQueue(ocl->queue), "clReleaseCommandQueue");
    check(clReleaseContext(ocl->ctx), "clReleaseContext");
  }
  nomp_free(&bnd->bptr);

  return 0;
}

static int opencl_device_query(nomp_backend_t *bnd, cl_device_id id) {
#define set_string(KEY, VAL)                                                   \
  {                                                                            \
    PyObject *obj = PyUnicode_FromString(VAL);                                 \
    PyDict_SetItemString(bnd->py_context, KEY, obj);                           \
    Py_XDECREF(obj);                                                           \
  }

#define set_int(KEY, VAL)                                                      \
  {                                                                            \
    PyObject *obj = PyLong_FromSize_t(VAL);                                    \
    PyDict_SetItemString(bnd->py_context, KEY, obj);                           \
    Py_XDECREF(obj);                                                           \
  }

  char val[BUFSIZ];
  check(clGetDeviceInfo(id, CL_DEVICE_NAME, sizeof(val), val, NULL),
        "clGetDeviceInfo");
  set_string("device::name", val);
  check(clGetDeviceInfo(id, CL_DEVICE_VENDOR, sizeof(val), val, NULL),
        "clGetDeviceInfo");
  set_string("device::vendor", val);
  check(clGetDeviceInfo(id, CL_DEVICE_VERSION, sizeof(val), val, NULL),
        "clGetDeviceInfo");
  set_string("device::driver", val);

  cl_device_type type;
  check(clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(type), &type, NULL),
        "clGetDeviceInfo");
  PyObject *obj = NULL;
  if (type & CL_DEVICE_TYPE_CPU) obj = PyUnicode_FromString("cpu");
  if (type & CL_DEVICE_TYPE_GPU) obj = PyUnicode_FromString("gpu");
  if (type & CL_DEVICE_TYPE_ACCELERATOR)
    obj = PyUnicode_FromString("accelerator");
  if (type & CL_DEVICE_TYPE_DEFAULT) obj = PyUnicode_FromString("default");
  PyDict_SetItemString(bnd->py_context, "device::type", obj);
  Py_XDECREF(obj);

  size_t max_threads_per_block;
  check(clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                        &max_threads_per_block, NULL),
        "clGetDeviceInfo");
  set_int("device::max_threads_per_block", max_threads_per_block);

#undef set_string
#undef set_int

  return 0;
}

/**
 * @ingroup nomp_backend_init
 * @brief Initializes OpenCL backend with the specified platform and device.
 *
 * Initializes OpenCL backend while creating a command queue using the
 * given platform id and device id. Returns a negative value if an error
 * occurred during the initialization, otherwise returns 0.
 *
 * @param[in] bnd Target backend for code generation.
 * @param[in] platform_id Target platform id.
 * @param[in] device_id Target device id.
 * @return int
 */
int opencl_init(nomp_backend_t *bnd, const int platform_id,
                const int device_id) {
  cl_uint num_platforms;
  check(clGetPlatformIDs(0, NULL, &num_platforms), "clGetPlatformIDs");
  if (platform_id < 0 || platform_id >= (int)num_platforms) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Platform id %d provided to libnomp is not valid.",
                    platform_id);
  }

  cl_platform_id *cl_platforms = nomp_calloc(cl_platform_id, num_platforms);
  check(clGetPlatformIDs(num_platforms, cl_platforms, &num_platforms),
        "clGetPlatformIDs");
  cl_platform_id platform = cl_platforms[platform_id];
  nomp_free(&cl_platforms);

  cl_uint num_devices;
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices),
        "clGetDeviceIDs");
  if (device_id < 0 || device_id >= (int)num_devices) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }

  cl_device_id *cl_devices = nomp_calloc(cl_device_id, num_devices);
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, cl_devices,
                       &num_devices),
        "clGetDeviceIDs");
  cl_device_id device = cl_devices[device_id];
  nomp_free(&cl_devices);

  nomp_check(opencl_device_query(bnd, device));

  struct opencl_backend_t *ocl = nomp_calloc(struct opencl_backend_t, 1);
  ocl->device_id               = device;
  cl_int err;
  ocl->ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  check(err, "clCreateContext");
  ocl->queue = clCreateCommandQueueWithProperties(ocl->ctx, device, 0, &err);
  check(err, "clCreateCommandQueueWithProperties");

  bnd->bptr      = (void *)ocl;
  bnd->update    = opencl_update;
  bnd->knl_build = opencl_knl_build;
  bnd->knl_run   = opencl_knl_run;
  bnd->knl_free  = opencl_knl_free;
  bnd->sync      = opencl_sync;
  bnd->finalize  = opencl_finalize;

  return 0;
}

#undef check
