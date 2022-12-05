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
      return set_log(NOMP_CUDA_FAILURE, NOMP_ERROR, ERR_STR_CUDA_FAILURE,      \
                     "operation", msg);                                        \
    }                                                                          \
  } while (0)

#define chk_cu(x) chk_cu_(__FILE__, __LINE__, x)

#define chk_nvrtc_(file, line, x)                                              \
  do {                                                                         \
    if (x != NVRTC_SUCCESS) {                                                  \
      const char *msg = nvrtcGetErrorString(x);                                \
      return set_log(NOMP_CUDA_FAILURE, NOMP_ERROR, ERR_STR_CUDA_FAILURE,      \
                     "runtime compilation", msg);                              \
    }                                                                          \
  } while (0)
#define chk_nvrtc(x) chk_nvrtc_(__FILE__, __LINE__, x)

#define chk_cuda_(file, line, x)                                               \
  do {                                                                         \
    if (x != cudaSuccess) {                                                    \
      const char *msg = cudaGetErrorString(x);                                 \
      return set_log(NOMP_CUDA_FAILURE, NOMP_ERROR, ERR_STR_CUDA_FAILURE,      \
                     "operation", msg);                                        \
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

static int cuda_update(struct backend *bnd, struct mem *m, const int op) {
  struct cuda_backend *ocl = (struct cuda_backend *)bnd->bptr;

  cudaError_t err;
  if (op & NOMP_ALLOC) {
    err = cudaMalloc(&m->bptr, (m->idx1 - m->idx0) * m->usize);
    chk_cuda(err);
  }

  if (op & NOMP_TO) {
    err = cudaMemcpy(m->bptr, m->hptr + m->usize * m->idx0,
                     (m->idx1 - m->idx0) * m->usize, cudaMemcpyHostToDevice);
    chk_cuda(err);
  }

  if (op == NOMP_FROM) {
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

static void cuda_update_ptr(void **p, size_t *size, struct mem *m) {
  *p = (void *)m->bptr;
  *size = sizeof(m->bptr);
}

static int cuda_knl_build(struct backend *bnd, struct prog *prg,
                          const char *source, const char *name) {
  nvrtcProgram prog;
  nvrtcResult nvrtc_err =
      nvrtcCreateProgram(&prog, source, NULL, 0, NULL, NULL);
  chk_nvrtc(nvrtc_err);

  struct cuda_backend *cbnd = (struct cuda_backend *)bnd->bptr;
  char arch[NOMP_BUFSIZ];
  snprintf(arch, NOMP_BUFSIZ, "-arch=compute_%d%d", cbnd->prop.major,
           cbnd->prop.minor);

  const char *opts[1] = {arch};
  nvrtc_err = nvrtcCompileProgram(prog, 1, opts);
  if (nvrtc_err != NVRTC_SUCCESS) {
    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    char *log = tcalloc(char, log_size);
    nvrtcGetProgramLog(prog, log);
    const char *err_str = nvrtcGetErrorString(nvrtc_err);
    size_t msg_size = log_size + strlen(err_str) + 2;
    char *msg = tcalloc(char, msg_size);
    snprintf(msg, msg_size, "%s: %s", err_str, log);
    int err_id = set_log(NOMP_CUDA_FAILURE, NOMP_ERROR, ERR_STR_CUDA_FAILURE,
                         "build", msg);
    tfree(log), tfree(msg);
    return err_id;
  }

  size_t ptx_size;
  nvrtc_err = nvrtcGetPTXSize(prog, &ptx_size);
  chk_nvrtc(nvrtc_err);

  char *ptx = tcalloc(char, ptx_size);
  nvrtc_err = nvrtcGetPTX(prog, ptx);
  chk_nvrtc(nvrtc_err);

  nvrtc_err = nvrtcDestroyProgram(&prog);
  chk_nvrtc(nvrtc_err);

  struct cuda_prog *cprg = prg->bptr = tcalloc(struct cuda_prog, 1);
  CUresult cu_err = cuModuleLoadData(&cprg->module, ptx);
  chk_cu(cu_err);

  tfree(ptx);

  cu_err = cuModuleGetFunction(&cprg->kernel, cprg->module, name);
  chk_cu(cu_err);

  return 0;
}

static int cuda_knl_run(struct backend *bnd, struct prog *prg, va_list args) {
  const int ndim = prg->ndim, narg = prg->narg;
  const size_t *global = prg->global, *local = prg->local;

  struct mem *m;
  void *vargs[NARGS_MAX];
  for (int i = 0; i < narg; i++) {
    void *p = va_arg(args, void *);

    int type = va_arg(args, int);
    switch (type) {
    case NOMP_INT:
    case NOMP_UINT:
    case NOMP_FLOAT:
      break;
    case NOMP_PTR:
      m = mem_if_mapped(p);
      if (m == NULL)
        return set_log(NOMP_USER_MAP_PTR_IS_INVALID, NOMP_ERROR,
                       ERR_STR_USER_MAP_PTR_IS_INVALID, p);
      p = &m->bptr;
      break;
    default:
      return set_log(NOMP_USER_KNL_ARG_TYPE_IS_INVALID, NOMP_ERROR,
                     ERR_STR_KNL_ARG_TYPE_IS_INVALID, type);
      break;
    }
    vargs[i] = p;
  }

  struct cuda_prog *cprg = (struct cuda_prog *)prg->bptr;
  int err = cuLaunchKernel(cprg->kernel, global[0], global[1], global[2],
                           local[0], local[1], local[2], 0, NULL, vargs, NULL);
  // FIXME: Wrong, call set_log.
  return err != CUDA_SUCCESS;
}

static int cuda_knl_free(struct prog *prg) {
  struct cuda_prog *cprg = (struct cuda_prog *)prg->bptr;
  int err = cuModuleUnload(cprg->module);
  // Check for error and call set_log
  return 0;
}

static int cuda_finalize(struct backend *bnd) {
  // Nothing to do
  return 0;
}

int cuda_init(struct backend *bnd, const int platform_id, const int device_id) {
  // Make sure a Context exists for nvrtc
  cudaFree(0);

  int num_devices;
  CUresult result = cudaGetDeviceCount(&num_devices);
  chk_cu(result);

  if (device_id < 0 || device_id >= num_devices) {
    return set_log(NOMP_USER_DEVICE_IS_INVALID, NOMP_ERROR,
                   ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }

  result = cudaSetDevice(device_id);
  chk_cu(result);

  struct cuda_backend *cuda = tcalloc(struct cuda_backend, 1);
  cuda->device_id = device_id;
  result = cudaGetDeviceProperties(&cuda->prop, device_id);
  chk_cu(result);

  bnd->bptr = (void *)cuda;
  bnd->update = cuda_update;
  bnd->knl_build = cuda_knl_build;
  bnd->knl_run = cuda_knl_run;
  bnd->knl_free = cuda_knl_free;
  bnd->finalize = cuda_finalize;

  return 0;
}

#undef NARGS_MAX
