#include "nomp-impl.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

static const char *ERR_STR_CUDA_FAILURE = "Cuda %s failed: %s.";

#define chk_cu(call)                                                           \
  {                                                                            \
    CUresult x = (call);                                                       \
    if (x != CUDA_SUCCESS) {                                                   \
      const char *msg;                                                         \
      cuGetErrorName(x, &msg);                                                 \
      return nomp_set_log(NOMP_CUDA_FAILURE, NOMP_ERROR, ERR_STR_CUDA_FAILURE, \
                          "CU operation", msg);                                \
    }                                                                          \
  }

#define chk_nvrtc(call)                                                        \
  {                                                                            \
    nvrtcResult x = (call);                                                    \
    if (x != NVRTC_SUCCESS) {                                                  \
      return nomp_set_log(NOMP_CUDA_FAILURE, NOMP_ERROR, ERR_STR_CUDA_FAILURE, \
                          "nvrtc operation", nvrtcGetErrorString(x));          \
    }                                                                          \
  }

#define chk_rt(call)                                                           \
  {                                                                            \
    cudaError_t x = (call);                                                    \
    if (x != cudaSuccess) {                                                    \
      return nomp_set_log(NOMP_CUDA_FAILURE, NOMP_ERROR, ERR_STR_CUDA_FAILURE, \
                          "runtime operation", cudaGetErrorString(x));         \
    }                                                                          \
  }

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

  if (op & NOMP_ALLOC)
    chk_rt(cudaMalloc(&m->bptr, (m->idx1 - m->idx0) * m->usize));

  if (op & NOMP_TO) {
    chk_rt(cudaMemcpy(m->bptr, m->hptr + m->usize * m->idx0,
                      (m->idx1 - m->idx0) * m->usize, cudaMemcpyHostToDevice));
  }

  if (op == NOMP_FROM) {
    chk_rt(cudaMemcpy(m->hptr + m->usize * m->idx0, m->bptr,
                      (m->idx1 - m->idx0) * m->usize, cudaMemcpyDeviceToHost));
  } else if (op == NOMP_FREE) {
    chk_rt(cudaFree(m->bptr));
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
  chk_nvrtc(nvrtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  struct cuda_backend *cuda = (struct cuda_backend *)bnd->bptr;
  char arch[MAX_BUFSIZ];
  snprintf(arch, MAX_BUFSIZ, "-arch=compute_%d%d", cuda->prop.major,
           cuda->prop.minor);

  const char *opts[1] = {arch};
  nvrtcResult result = nvrtcCompileProgram(prog, 1, opts);
  if (result != NVRTC_SUCCESS) {
    size_t size;
    chk_nvrtc(nvrtcGetProgramLogSize(prog, &size));
    char *log = nomp_calloc(char, size + 1);
    chk_nvrtc(nvrtcGetProgramLog(prog, log));

    const char *err = nvrtcGetErrorString(result);
    size += strlen(err) + 2 + 1;

    char *msg = nomp_calloc(char, size);
    snprintf(msg, size, "%s: %s", err, log);

    int id = nomp_set_log(NOMP_CUDA_FAILURE, NOMP_ERROR, ERR_STR_CUDA_FAILURE,
                          "build", msg);
    nomp_free(log), nomp_free(msg);
    return id;
  }

  size_t ptx_size;
  chk_nvrtc(nvrtcGetPTXSize(prog, &ptx_size));
  char *ptx = nomp_calloc(char, ptx_size);
  chk_nvrtc(nvrtcGetPTX(prog, ptx));

  struct cuda_prog *cprg = prg->bptr = nomp_calloc(struct cuda_prog, 1);
  chk_cu(cuModuleLoadData(&cprg->module, ptx));
  chk_cu(cuModuleGetFunction(&cprg->kernel, cprg->module, name));

  nomp_free(ptx);
  chk_nvrtc(nvrtcDestroyProgram(&prog));

  return 0;
}

static int cuda_knl_run(struct backend *bnd, struct prog *prg, va_list args) {
  void *vargs[MAX_KNL_ARGS];
  for (int i = 0; i < prg->nargs; i++) {
    const char *var = va_arg(args, const char *);
    int type = va_arg(args, int);
    size_t size = va_arg(args, size_t);
    void *p = va_arg(args, void *);

    struct mem *m;
    switch (type) {
    case NOMP_INT:
    case NOMP_FLOAT:
      break;
    case NOMP_PTR:
      m = mem_if_mapped(p);
      if (m == NULL)
        return nomp_set_log(NOMP_USER_MAP_PTR_IS_INVALID, NOMP_ERROR,
                            ERR_STR_USER_MAP_PTR_IS_INVALID, p);
      p = &m->bptr;
      break;
    default:
      return nomp_set_log(NOMP_USER_KNL_ARG_TYPE_IS_INVALID, NOMP_ERROR,
                          "Invalid libnomp kernel argument type %d.", type);
      break;
    }
    vargs[i] = p;
  }

  const size_t *global = prg->global, *local = prg->local;
  struct cuda_prog *cprg = (struct cuda_prog *)prg->bptr;
  chk_cu(cuLaunchKernel(cprg->kernel, global[0], global[1], global[2], local[0],
                        local[1], local[2], 0, NULL, vargs, NULL));
}

static int cuda_knl_free(struct prog *prg) {
  struct cuda_prog *cprg = (struct cuda_prog *)prg->bptr;
  chk_cu(cuModuleUnload(cprg->module));
  return 0;
}

static int cuda_sync(struct backend *bnd) { chk_rt(cudaDeviceSynchronize()); }

static int cuda_finalize(struct backend *bnd) {
  // Nothing to do
  return 0;
}

int cuda_init(struct backend *bnd, const int platform_id, const int device_id) {
  // Make sure a context exists for nvrtc.
  cudaFree(0);

  int num_devices;
  chk_cu(cudaGetDeviceCount(&num_devices));

  if (device_id < 0 || device_id >= num_devices) {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }

  chk_rt(cudaSetDevice(device_id));

  struct cuda_backend *cuda = bnd->bptr = nomp_calloc(struct cuda_backend, 1);
  cuda->device_id = device_id;
  chk_rt(cudaGetDeviceProperties(&cuda->prop, device_id));

  bnd->update = cuda_update;
  bnd->knl_build = cuda_knl_build;
  bnd->knl_run = cuda_knl_run;
  bnd->knl_free = cuda_knl_free;
  bnd->sync = cuda_sync;
  bnd->finalize = cuda_finalize;

  return 0;
}
