#include "nomp-impl.h"

#define CL_TARGET_OPENCL_VERSION 220
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

static const char *ERR_STR_OPENCL_FAILURE = "%s failed with error code: %d.";

#define chk_cl(call, msg)                                                      \
  {                                                                            \
    cl_int err = (call);                                                       \
    if (err != CL_SUCCESS) {                                                   \
      return nomp_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE, \
                      msg, err);                                               \
    }                                                                          \
  }

struct opencl_backend {
  cl_device_id device_id;
  cl_command_queue queue;
  cl_context ctx;
};

struct opencl_prog {
  cl_program prg;
  cl_kernel knl;
};

static int opencl_update(struct nomp_backend_t *bnd, struct nomp_mem_t *m,
                         const nomp_map_direction_t op, size_t start,
                         size_t end, size_t usize) {
  struct opencl_backend *ocl = (struct opencl_backend *)bnd->bptr;

  cl_int err;
  if (op & NOMP_ALLOC) {
    cl_mem *clm = nomp_calloc(cl_mem, 1);
    *clm = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE,
                          NOMP_MEM_BYTES(start, end, usize), NULL, &err);
    chk_cl(err, "clCreateBuffer");
    m->bptr = (void *)clm, m->bsize = sizeof(cl_mem);
  }

  cl_mem *clm = (cl_mem *)m->bptr;
  if (op & NOMP_TO) {
    chk_cl(clEnqueueWriteBuffer(ocl->queue, *clm, CL_TRUE,
                                NOMP_MEM_OFFSET(start - m->idx0, usize),
                                NOMP_MEM_BYTES(start, end, usize),
                                (char *)m->hptr + NOMP_MEM_OFFSET(start, usize),
                                0, NULL, NULL),
           "clEnqueueWriteBuffer");
  } else if (op == NOMP_FROM) {
    chk_cl(clEnqueueReadBuffer(ocl->queue, *clm, CL_TRUE,
                               NOMP_MEM_OFFSET(start - m->idx0, usize),
                               NOMP_MEM_BYTES(start, end, usize),
                               (char *)m->hptr + NOMP_MEM_OFFSET(start, usize),
                               0, NULL, NULL),
           "clEnqueueReadBuffer");
  } else if (op == NOMP_FREE) {
    chk_cl(clReleaseMemObject(*clm), "clReleaseMemObject");
    nomp_free(&m->bptr);
  }

  return 0;
}

static int opencl_knl_build(struct nomp_backend_t *bnd, struct nomp_prog_t *prg,
                            const char *source, const char *name) {
  struct opencl_backend *ocl = bnd->bptr;
  struct opencl_prog *ocl_prg = prg->bptr = nomp_calloc(struct opencl_prog, 1);

  cl_int err;
  ocl_prg->prg = clCreateProgramWithSource(
      ocl->ctx, 1, (const char **)(&source), NULL, &err);
  chk_cl(err, "clCreateProgramWithSource");

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
  chk_cl(err, "clCreateKernel");

  return 0;
}

static int opencl_knl_run(struct nomp_backend_t *bnd, struct nomp_prog_t *prg) {
  struct opencl_prog *oprg = (struct opencl_prog *)prg->bptr;

  for (int i = 0; i < prg->nargs; i++) {
    chk_cl(clSetKernelArg(oprg->knl, i, prg->args[i].size, prg->args[i].ptr),
           "clSetKernelArg");
  }

  struct opencl_backend *ocl = (struct opencl_backend *)bnd->bptr;
  chk_cl(clEnqueueNDRangeKernel(ocl->queue, oprg->knl, prg->ndim, NULL,
                                prg->gws, prg->local, 0, NULL, NULL),
         "clEnqueueNDRangeKernel");

  return 0;
}

static int opencl_knl_free(struct nomp_prog_t *prg) {
  struct opencl_prog *ocl_prg = prg->bptr;

  if (ocl_prg) {
    chk_cl(clReleaseKernel(ocl_prg->knl), "clReleaseKernel");
    chk_cl(clReleaseProgram(ocl_prg->prg), "clReleaseProgram");
  }

  nomp_free(&prg->bptr);
  return 0;
}

static int opencl_sync(struct nomp_backend_t *bnd) {
  struct opencl_backend *ocl = (struct opencl_backend *)bnd->bptr;

  chk_cl(clFinish(ocl->queue), "clFinish");

  return 0;
}

static int opencl_finalize(struct nomp_backend_t *bnd) {
  struct opencl_backend *ocl = bnd->bptr;

  if (ocl) {
    chk_cl(clReleaseCommandQueue(ocl->queue), "clReleaseCommandQueue");
    chk_cl(clReleaseContext(ocl->ctx), "clReleaseContext");
  }
  nomp_free(&bnd->bptr);

  return 0;
}

static int opencl_device_query(struct nomp_backend_t *bnd, cl_device_id id) {
  bnd->py_context = PyDict_New();

#define set_string_aux(KEY, VAL)                                               \
  {                                                                            \
    PyObject *obj = PyUnicode_FromString(VAL);                                 \
    PyDict_SetItemString(bnd->py_context, KEY, obj);                           \
    Py_XDECREF(obj);                                                           \
  }

  set_string_aux("backend::name", "opencl");

#define set_string_info(PARAM, KEY)                                            \
  {                                                                            \
    char string[BUFSIZ];                                                       \
    chk_cl(clGetDeviceInfo(id, PARAM, sizeof(string), string, NULL),           \
           "clGetDeviceInfo");                                                 \
    set_string_aux(KEY, string);                                               \
  }

  set_string_info(CL_DEVICE_NAME, "device::name");
  set_string_info(CL_DEVICE_VENDOR, "device::vendor");
  set_string_info(CL_DRIVER_VERSION, "device::driver");

#undef set_string_info
#undef set_string_aux

  cl_device_type type;
  chk_cl(clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(type), &type, NULL),
         "clGetDeviceInfo");
  PyObject *obj = NULL;
  if (type & CL_DEVICE_TYPE_CPU)
    obj = PyUnicode_FromString("cpu");
  if (type & CL_DEVICE_TYPE_GPU)
    obj = PyUnicode_FromString("gpu");
  if (type & CL_DEVICE_TYPE_ACCELERATOR)
    obj = PyUnicode_FromString("accelerator");
  if (type & CL_DEVICE_TYPE_DEFAULT)
    obj = PyUnicode_FromString("default");
  PyDict_SetItemString(bnd->py_context, "device::type", obj);
  Py_XDECREF(obj);

#define set_int_info(T, PARAM, KEY)                                            \
  {                                                                            \
    T value;                                                                   \
    chk_cl(clGetDeviceInfo(id, PARAM, sizeof(T), &value, NULL),                \
           "clGetDeviceInfo");                                                 \
    PyObject *obj = PyLong_FromSize_t(value);                                  \
    PyDict_SetItemString(bnd->py_context, KEY, obj);                           \
    Py_XDECREF(obj);                                                           \
  }

  set_int_info(cl_uint, CL_DEVICE_MAX_COMPUTE_UNITS,
               "device::max_compute_units");
  set_int_info(size_t, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
               "device::max_work_item_dims");
  set_int_info(size_t, CL_DEVICE_MAX_WORK_GROUP_SIZE,
               "device::max_work_group_size");
  set_int_info(cl_uint, CL_DEVICE_MAX_CLOCK_FREQUENCY,
               "device::max_clock_frequency");

#undef set_int_info

  return 0;
}

int opencl_init(struct nomp_backend_t *bnd, const int platform_id,
                const int device_id) {
  cl_uint num_platforms;
  cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (platform_id < 0 | platform_id >= num_platforms) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Platform id %d provided to libnomp is not valid.",
                    platform_id);
  }

  cl_platform_id *cl_platforms = nomp_calloc(cl_platform_id, num_platforms);
  err = clGetPlatformIDs(num_platforms, cl_platforms, &num_platforms);
  cl_platform_id platform = cl_platforms[platform_id];
  nomp_free(&cl_platforms);

  cl_uint num_devices;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  if (device_id < 0 || device_id >= num_devices) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }

  cl_device_id *cl_devices = nomp_calloc(cl_device_id, num_devices);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, cl_devices,
                       &num_devices);
  cl_device_id device = cl_devices[device_id];
  nomp_free(&cl_devices);

  nomp_check(opencl_device_query(bnd, device));

  struct opencl_backend *ocl = nomp_calloc(struct opencl_backend, 1);
  ocl->device_id = device;
  ocl->ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  ocl->queue = clCreateCommandQueueWithProperties(ocl->ctx, device, 0, &err);

  bnd->bptr = (void *)ocl;
  bnd->update = opencl_update;
  bnd->knl_build = opencl_knl_build;
  bnd->knl_run = opencl_knl_run;
  bnd->knl_free = opencl_knl_free;
  bnd->sync = opencl_sync;
  bnd->finalize = opencl_finalize;

  return 0;
}
