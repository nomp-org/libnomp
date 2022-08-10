#include "nomp-impl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#define NARGS_MAX 64

static int cuda_map(struct backend *bnd, struct mem *m, const int op);
static int cuda_knl_build(struct backend *bnd, struct prog *prg,
                          const char *source, const char *name);
static int cuda_knl_run(struct backend *bnd, struct prog *prg, const int ndim,
                        const size_t *global, const size_t *local, int nargs,
                        va_list args);
static int cuda_knl_free(struct prog *prg);
static int cuda_finalize(struct backend *bnd);

struct cuda_backend {
  int device_id;
  struct cudaDeviceProp prop;
};

int cuda_init(struct backend *bnd, const int platform_id, const int device_id) {
  int num_devices;
  int ierr = cudaGetDeviceCount(&num_devices);
  if (device_id < 0 || device_id >= num_devices)
    return NOMP_INVALID_DEVICE;

  struct cuda_backend *cuda = bnd->bptr =
      calloc(1, sizeof(struct cuda_backend));

  bnd->map = cuda_map;
  bnd->knl_build = cuda_knl_build;
  bnd->knl_run = cuda_knl_run;
  bnd->knl_free = cuda_knl_free;
  bnd->finalize = cuda_finalize;

  return 0;
}

static int cuda_map(struct backend *bnd, struct mem *m, const int op) {
  struct cuda_backend *ocl = bnd->bptr;

  int err;
  if (op & NOMP_ALLOC) {
    err = cudaMalloc(&m->bptr, (m->idx1 - m->idx0) * m->usize);
    if (err != cudaSuccess)
      return 1;
  }

  if (op & NOMP_H2D) {
    err = cudaMemcpy(m->bptr, m->hptr + m->usize * m->idx0,
                     (m->idx1 - m->idx0) * m->usize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      return 1;
  }

  if (op == NOMP_D2H) {
    err = cudaMemcpy(m->hptr + m->usize * m->idx0, m->bptr,
                     (m->idx1 - m->idx0) * m->usize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      return 1;
  } else if (op == NOMP_FREE) {
    err = cudaFree(m->bptr);
    if (err != cudaSuccess)
      return 1;
    m->bptr = NULL;
  }

  return 0;
}

static void cuda_map_ptr(void **p, size_t *size, struct mem *m) {
  *p = (void *)m->bptr;
  *size = sizeof(m->bptr);
}

struct cuda_prog {
  CUmodule module;
  CUfunction kernel;
};

static int cuda_knl_build(struct backend *bnd, struct prog *prg,
                          const char *source, const char *name) {
  nvrtcProgram prog;
  int err = nvrtcCreateProgram(&prog, source, NULL, 0, NULL, NULL);
  if (err != NVRTC_SUCCESS)
    return 1;

  struct cuda_backend *cbnd = (struct cuda_backend *)bnd->bptr;
  char arch[BUFSIZ];
  snprintf(arch, BUFSIZ, "-arch=compute_%d%d", cbnd->prop.major,
           cbnd->prop.minor);

  const char *opts[2] = {"-default-device", arch};
  err = nvrtcCompileProgram(prog, 2, opts);
  if (err != NVRTC_SUCCESS) {
    // TODO: Get ther error log
    return 1;
  }

  size_t ptx_size;
  err = nvrtcGetPTXSize(prog, &ptx_size);
  if (err != NVRTC_SUCCESS)
    return 1;
  char *ptx = (char *)calloc(ptx_size, sizeof(char));
  err = nvrtcGetPTX(prog, ptx);
  if (err != NVRTC_SUCCESS)
    return 1;
  err = nvrtcDestroyProgram(&prog);
  if (err != NVRTC_SUCCESS)
    return 1;

  struct cuda_prog *cprg = prg->bptr = calloc(1, sizeof(struct cuda_prog));
  err = cuModuleLoadData(&cprg->module, ptx);
  if (err != NVRTC_SUCCESS)
    return 1;
  if (ptx)
    free(ptx);
  err = cuModuleGetFunction(&cprg->kernel, cprg->module, name);

  return 0;
}

static int cuda_knl_run(struct backend *bnd, struct prog *prg, const int ndim,
                        const size_t *global, const size_t *local, int nargs,
                        va_list args) {
  size_t size;
  struct mem *m;
  void *vargs[NARGS_MAX];
  for (int i = 0; i < nargs; i++) {
    int type = va_arg(args, int);
    void *p = va_arg(args, void *);
    switch (type) {
    case NOMP_INTEGER:
    case NOMP_FLOAT:
      size = va_arg(args, size_t);
      break;
    case NOMP_PTR:
      m = mem_if_mapped(p);
      if (m == NULL)
        return NOMP_INVALID_MAP_PTR;
      p = m->bptr;
      break;
    default:
      return NOMP_KNL_ARG_TYPE_ERROR;
      break;
    }
    vargs[i] = p;
  }

  struct cuda_prog *cprg = (struct cuda_prog *)prg->bptr;
  int err = cuLaunchKernel(cprg->kernel, global[0], global[1], global[2],
                           local[0], local[1], local[2], 0, NULL, vargs, NULL);
  return err != CUDA_SUCCESS;
}

static int cuda_knl_free(struct prog *prg) {
  struct cuda_prog *cprg = (struct cuda_prog *)prg->bptr;
  int err = cuModuleUnload(cprg->module);
  return 0;
}

static int cuda_finalize(struct backend *bnd) {
  // Nothing to do
  return 0;
}
