#include "nomp-impl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#define NARGS_MAX 64

#define chk_cu_(file, line, x)                                                 \
  do {                                                                         \
    if (x != CUDA_SUCCESS) {                                                   \
      const char *msg;                                                         \
      cuGetErrorName(x, &msg);                                                 \
      fprintf(stderr, "%s:%u %s\n", file, line, msg);                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define chk_cu(x) chk_cu_(__FILE__, __LINE__, x)

#define chk_nvrtc_(file, line, x)                                              \
  do {                                                                         \
    if (x != NVRTC_SUCCESS) {                                                  \
      fprintf(stderr, "%s:%d %s\n", file, line, nvrtcGetErrorString(x));       \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define chk_nvrtc(x) chk_nvrtc_(__FILE__, __LINE__, x)

#define chk_cuda_(file, line, x)                                               \
  do {                                                                         \
    if (x != cudaSuccess) {                                                    \
      fprintf(stderr, "%s:%d %s\n", file, line, cudaGetErrorString(x));        \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define chk_cuda(x) chk_cuda_(__FILE__, __LINE__, x)

struct cuda_backend {
  int device_id;
  struct cudaDeviceProp prop;
};

struct cuda_prog {
  CUmodule module;
  CUfunction kernel;
};

static int cuda_map(struct backend *bnd, struct mem *m, const int op) {
  struct cuda_backend *ocl = (struct cuda_backend *)bnd->bptr;

  cudaError_t err;
  if (op & NOMP_ALLOC) {
    err = cudaMalloc(&m->bptr, (m->idx1 - m->idx0) * m->usize);
    chk_cuda(err);
  }

  if (op & NOMP_H2D) {
    err = cudaMemcpy(m->bptr, m->hptr + m->usize * m->idx0,
                     (m->idx1 - m->idx0) * m->usize, cudaMemcpyHostToDevice);
    chk_cuda(err);
  }

  if (op == NOMP_D2H) {
    err = cudaMemcpy(m->hptr + m->usize * m->idx0, m->bptr,
                     (m->idx1 - m->idx0) * m->usize, cudaMemcpyDeviceToHost);
    chk_cuda(err);
  } else if (op == NOMP_FREE) {
    err = cudaFree(m->bptr);
    chk_cuda(err);
    m->bptr = NULL;
  }

  return 0;
}

static void cuda_map_ptr(void **p, size_t *size, struct mem *m) {
  *p = (void *)m->bptr;
  *size = sizeof(m->bptr);
}

static int cuda_knl_build(struct backend *bnd, struct prog *prg,
                          const char *source, const char *name) {
  nvrtcProgram prog;
  nvrtcResult nvrtc_err =
      nvrtcCreateProgram(&prog, source, "foo.cu", 0, NULL, NULL);
  chk_nvrtc(nvrtc_err);

  struct cuda_backend *cbnd = (struct cuda_backend *)bnd->bptr;
  char arch[BUFSIZ];
  snprintf(arch, BUFSIZ, "-arch=compute_%d%d", cbnd->prop.major,
           cbnd->prop.minor);

  const char *opts[1] = {arch};
  nvrtc_err = nvrtcCompileProgram(prog, 1, opts);
  if (nvrtc_err != NVRTC_SUCCESS) {
    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    char *log = (char *)calloc(log_size, sizeof(char));
    nvrtcGetProgramLog(prog, log);
    fprintf(stderr, "%s\n%s", nvrtcGetErrorString(nvrtc_err), log);
    free(log);
  }

  size_t ptx_size;
  nvrtc_err = nvrtcGetPTXSize(prog, &ptx_size);
  chk_nvrtc(nvrtc_err);

  char *ptx = (char *)calloc(ptx_size, sizeof(char));
  nvrtc_err = nvrtcGetPTX(prog, ptx);
  chk_nvrtc(nvrtc_err);

  nvrtc_err = nvrtcDestroyProgram(&prog);
  chk_nvrtc(nvrtc_err);

  struct cuda_prog *cprg = prg->bptr = calloc(1, sizeof(struct cuda_prog));
  CUresult cu_err = cuModuleLoadData(&cprg->module, ptx);
  chk_cu(cu_err);

  if (ptx)
    free(ptx);

  cu_err = cuModuleGetFunction(&cprg->kernel, cprg->module, name);
  chk_cu(cu_err);

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
      p = (void *)m->bptr;
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

int cuda_init(struct backend *bnd, const int platform_id, const int device_id) {
  cudaFree(0); // Make sure a Context exists for nvrtc
  int num_devices;
  CUresult result = cudaGetDeviceCount(&num_devices);
  chk_cu(result);
  if (device_id < 0 || device_id >= num_devices)
    return NOMP_INVALID_DEVICE;
  result = cudaSetDevice(device_id);
  chk_cu(result);

  struct cuda_backend *cbnd = bnd->bptr =
      calloc(1, sizeof(struct cuda_backend));
  cbnd->device_id = device_id;
  result = cudaGetDeviceProperties(&cbnd->prop, device_id);
  chk_cu(result);

  bnd->map = cuda_map;
  bnd->knl_build = cuda_knl_build;
  bnd->knl_run = cuda_knl_run;
  bnd->knl_free = cuda_knl_free;
  bnd->finalize = cuda_finalize;

  return 0;
}

#undef NARGS_MAX
