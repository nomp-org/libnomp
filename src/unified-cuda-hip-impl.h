#include "nomp-impl.h"

#define TOKEN_PASTE_(a, b) a##b
#define TOKEN_PASTE(a, b) TOKEN_PASTE_(a, b)

#define gpu_update TOKEN_PASTE(GPU, _update)
#define gpu_update_ptr TOKEN_PASTE(GPU, _update_ptr)
#define gpu_knl_build TOKEN_PASTE(GPU, _knl_build)
#define gpu_knl_run TOKEN_PASTE(GPU, _knl_run)
#define gpu_knl_free TOKEN_PASTE(GPU, _knl_free)
#define gpu_sync TOKEN_PASTE(GPU, _sync)
#define gpu_finalize TOKEN_PASTE(GPU, _finalize)
#define gpu_init TOKEN_PASTE(GPU, _init)
#define gpu_backend TOKEN_PASTE(GPU, _backend)
#define gpu_prog TOKEN_PASTE(GPU, _prog)

#define gpuError_t TOKEN_PASTE(GPU, Error_t)
#define gpuSuccess TOKEN_PASTE(GPU, Success)
#define gpuGetErrorName TOKEN_PASTE(GPU, GetErrorName)
#define gpuMalloc TOKEN_PASTE(GPU, Malloc)
#define gpuMemcpy TOKEN_PASTE(GPU, Memcpy)
#define gpuFree TOKEN_PASTE(GPU, Free)
#define gpuMemcpyHostToDevice TOKEN_PASTE(GPU, MemcpyHostToDevice)
#define gpuMemcpyDeviceToHost TOKEN_PASTE(GPU, MemcpyDeviceToHost)
#define gpuDeviceSynchronize TOKEN_PASTE(GPU, DeviceSynchronize)
#define gpuGetDeviceCount TOKEN_PASTE(GPU, GetDeviceCount)
#define gpuSetDevice TOKEN_PASTE(GPU, SetDevice)
#define gpuGetDeviceProperties TOKEN_PASTE(GPU, GetDeviceProperties)

#define gpurtcResult TOKEN_PASTE(RUNTIME, Result)
#define gpurtcGetErrorString TOKEN_PASTE(RUNTIME, GetErrorString)
#define gpurtcProgram TOKEN_PASTE(RUNTIME, Program)
#define gpurtcCreateProgram TOKEN_PASTE(RUNTIME, CreateProgram)
#define gpurtcCompileProgram TOKEN_PASTE(RUNTIME, CompileProgram)
#define gpurtcGetProgramLogSize TOKEN_PASTE(RUNTIME, GetProgramLogSize)
#define gpurtcGetProgramLog TOKEN_PASTE(RUNTIME, GetProgramLog)
#define gpurtcDestroyProgram TOKEN_PASTE(RUNTIME, DestroyProgram)

#define chk_gpu_(file, line, call)                                             \
  do {                                                                         \
    gpuError_t result = (call);                                                \
    if (result != gpuSuccess) {                                                \
      const char *msg = gpuGetErrorName(result);                               \
      return nomp_set_log(NOMP_GPU_FAILURE, NOMP_ERROR, ERR_STR_GPU_FAILURE,   \
                          "operation", msg);                                   \
    }                                                                          \
  } while (0)

#define chk_gpu(call) chk_gpu_(__FILE__, __LINE__, call)

#define chk_gpurtc_(file, line, call)                                          \
  do {                                                                         \
    gpurtcResult result = (call);                                              \
    if (result != GPURTC_SUCCESS) {                                            \
      const char *msg = gpurtcGetErrorString(result);                          \
      return nomp_set_log(NOMP_GPU_FAILURE, NOMP_ERROR, ERR_STR_GPU_FAILURE,   \
                          "runtime compilation", msg);                         \
    }                                                                          \
  } while (0)

#define chk_gpurtc(call) chk_gpurtc_(__FILE__, __LINE__, call)

struct gpu_backend {
  int device_id;
  struct gpuDeviceProp prop;
  gpuCtx ctx;
};

struct gpu_prog {
  gpuModule module;
  gpuFunction kernel;
};

static int gpu_update(struct backend *bnd, struct mem *m, const int op) {
  if (op & NOMP_ALLOC)
    chk_gpu(gpuMalloc(&m->bptr, (m->idx1 - m->idx0) * m->usize));

  if (op & NOMP_TO) {
    chk_gpu(gpuMemcpy(m->bptr, (char *)m->hptr + m->usize * m->idx0,
                      (m->idx1 - m->idx0) * m->usize, gpuMemcpyHostToDevice));
  }

  if (op == NOMP_FROM) {
    chk_gpu(gpuMemcpy((char *)m->hptr + m->usize * m->idx0, m->bptr,
                      (m->idx1 - m->idx0) * m->usize, gpuMemcpyDeviceToHost));
  } else if (op == NOMP_FREE) {
    chk_gpu(gpuFree(m->bptr));
    m->bptr = NULL;
  }

  return 0;
}

static void gpu_update_ptr(void **p, size_t *size, struct mem *m) {
  *p = (void *)m->bptr;
  *size = sizeof(m->bptr);
}

static int gpu_knl_build(struct backend *bnd, struct prog *prg,
                         const char *source, const char *name) {
  struct gpu_backend *nbnd = (struct gpu_backend *)bnd->bptr;

  gpurtcProgram prog;
  chk_gpurtc(gpurtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  GPU_COMPILE
  if (result != GPURTC_SUCCESS) {
    size_t size;
    gpurtcGetProgramLogSize(prog, &size);
    char *log = nomp_calloc(char, size + 1);
    gpurtcGetProgramLog(prog, log);
    const char *err_str = gpurtcGetErrorString(result);
    size += strlen(err_str) + 2 + 1;

    char *msg = nomp_calloc(char, size);
    snprintf(msg, size, "%s: %s", err_str, log);
    int err_id = nomp_set_log(NOMP_GPU_FAILURE, NOMP_ERROR, ERR_STR_GPU_FAILURE,
                              "build", msg);
    nomp_free(log), nomp_free(msg);
    return err_id;
  }

  size_t code_size;
  chk_gpurtc(gpurtcGetCodeSize(prog, &code_size));
  char *code = nomp_calloc(char, code_size);
  chk_gpurtc(gpurtcGetCode(prog, code));

  struct gpu_prog *nprg = prg->bptr = nomp_calloc(struct gpu_prog, 1);
  GPU_CHECK(gpuModuleLoadData(&nprg->module, code));
  GPU_CHECK(gpuModuleGetFunction(&nprg->kernel, nprg->module, name));

  nomp_free(code);
  chk_gpurtc(gpurtcDestroyProgram(&prog));

  return 0;
}

static int gpu_knl_run(struct backend *bnd, struct prog *prg) {
  void *vargs[MAX_KNL_ARGS];
  struct arg *args = prg->args;
  for (int i = 0; i < prg->nargs; i++) {
    if (args[i].type == NOMP_PTR)
      vargs[i] = &args[i].ptr;
    else
      vargs[i] = args[i].ptr;
  }

  const size_t *global = prg->global, *local = prg->local;
  struct gpu_prog *cprg = (struct gpu_prog *)prg->bptr;
  GPU_CHECK(gpuModuleLaunchKernel(cprg->kernel, global[0], global[1], global[2],
                                  local[0], local[1], local[2], 0, NULL, vargs,
                                  NULL));
  return 0;
}

static int gpu_knl_free(struct prog *prg) {
  struct gpu_prog *cprg = (struct gpu_prog *)prg->bptr;
  GPU_CHECK(gpuModuleUnload(cprg->module));
  return 0;
}

static int gpu_sync(struct backend *bnd) {
  chk_gpu(gpuDeviceSynchronize());
  return 0;
}

static int gpu_finalize(struct backend *bnd) {
#ifndef __HIP_PLATFORM_HCC__
  struct gpu_backend *nbnd = (struct gpu_backend *)bnd->bptr;
  GPU_CHECK(gpuCtxDestroy(nbnd->ctx));
#endif
  return 0;
}

int gpu_init(struct backend *bnd, const int platform_id, const int device_id) {
  gpuFree(0);

  int num_devices;
  GPU_CHECK(gpuGetDeviceCount(&num_devices));
  if (device_id < 0 || device_id >= num_devices) {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }

  chk_gpu(gpuSetDevice(device_id));

  struct gpu_backend *nbnd = nomp_calloc(struct gpu_backend, 1);
  nbnd->device_id = device_id;
  chk_gpu(gpuGetDeviceProperties(&nbnd->prop, device_id));

#ifndef __HIP_PLATFORM_HCC__
  GPU_CHECK(gpuInit(0));
  GPU_CHECK(gpuCtxCreate(&nbnd->ctx, 0, nbnd->device_id));
#endif
  bnd->bptr = (void *)nbnd;

  bnd->update = gpu_update;
  bnd->knl_build = gpu_knl_build;
  bnd->knl_run = gpu_knl_run;
  bnd->knl_free = gpu_knl_free;
  bnd->sync = gpu_sync;
  bnd->finalize = gpu_finalize;

  return 0;
}

#undef TOKEN_PASTE_
#undef TOKEN_PASTE

#undef gpu_update
#undef gpu_update_ptr
#undef gpu_knl_build
#undef gpu_knl_run
#undef gpu_knl_free
#undef gpu_sync
#undef gpu_finalize
#undef gpu_init
#undef gpu_backend
#undef gpu_prog

#undef gpuError_t
#undef gpuSuccess
#undef gpuGetErrorName
#undef gpuMalloc
#undef gpuMemcpy
#undef gpuFree
#undef gpuMemcpyHostToDevice
#undef gpuMemcpyDeviceToHost
#undef gpuDeviceSynchronize
#undef gpuGetDeviceCount
#undef gpuSetDevice
#undef gpuGetDeviceProperties

#undef gpurtcResult
#undef gpurtcGetErrorString
#undef gpurtcProgram
#undef gpurtcCreateProgram
#undef gpurtcCompileProgram
#undef gpurtcGetProgramLogSize
#undef gpurtcGetProgramLog
#undef gpurtcDestroyProgram
