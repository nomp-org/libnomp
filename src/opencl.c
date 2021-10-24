#include <gnomp-impl.h>

int opencl_init(struct backend *ocl, int platform_id, int device_id) {
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

  ocl->ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  ocl->queue = clCreateCommandQueueWithProperties(ocl->ctx, device, NULL, &err);
  ocl->backend = GNOMP_OCL;

  free(cl_devices);
  free(cl_platforms);

  return 0;
}

int opencl_map(struct backend *ocl, struct mem *m, void *ptr, size_t id0,
               size_t id1, size_t usize, int direction, int alloc) {
  cl_int err;
  if (alloc) {
    m->size = id1 - id0;
    m->usize = usize;
    m->h_ptr = ptr;
    m->d_ptr = clCreateBuffer(ocl->ctx, CL_MEM_READ_WRITE, (id1 - id0) * usize,
                              NULL, &err);
  }
  if (err != CL_SUCCESS)
    return 1;

  // copy the content now
  if (direction == GNOMP_H2D)
    err = clEnqueueWriteBuffer(ocl->queue, m->d_ptr, CL_TRUE, 0,
                               (id1 - id0) * usize, m->h_ptr, 0, NULL, NULL);
  else if (direction == GNOMP_D2H)
    err = clEnqueueReadBuffer(ocl->queue, m->d_ptr, CL_TRUE, 0,
                              (id1 - id0) * usize, m->h_ptr, 0, NULL, NULL);

  if (err != CL_SUCCESS)
    return 1;

  return 0;
}

int opencl_build_knl(struct backend *ocl, struct prog *prg, const char *source,
                     const char *name) {
  cl_int err;
  prg->prg = clCreateProgramWithSource(ocl->ctx, 1, (const char **)(&source),
                                       NULL, &err);
  if (err != CL_SUCCESS)
    return 1;

  err = clBuildProgram(prg->prg, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    prg->prg = NULL;
    prg->knl = NULL;
    return 1;
  }

  prg->knl = clCreateKernel(prg->prg, name, &err);
  if (err != CL_SUCCESS) {
    prg->knl = NULL;
    return 1;
  }

  return 0;
}

int opencl_run_knl(struct backend *ocl, struct prog *prg, int nargs,
                   va_list args) {
  return 0;
}
