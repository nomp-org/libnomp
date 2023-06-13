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
#define gpu_compile TOKEN_PASTE(GPU, _compile)

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

#define chk_err_(file, line, call, err_t, success_t, retrive_err, type)        \
  do {                                                                         \
    err_t result = (call);                                                     \
    if (result != success_t) {                                                 \
      const char *msg;                                                         \
      retrive_err;                                                             \
      return nomp_log(NOMP_GPU_FAILURE, NOMP_ERROR, ERR_STR_GPU_FAILURE, type, \
                      msg);                                                    \
    }                                                                          \
  } while (0)

#define chk_gpu(call)                                                          \
  chk_err_(__FILE__, __LINE__, call, gpuError_t, gpuSuccess,                   \
           msg = gpuGetErrorName(result), "operation");

#define chk_gpurtc(call)                                                       \
  chk_err_(__FILE__, __LINE__, call, gpurtcResult, GPURTC_SUCCESS,             \
           msg = gpurtcGetErrorString(result), "runtime compilation")

struct gpu_backend {
  int device_id;
  struct gpuDeviceProp prop;
  gpuCtx ctx;
};

struct gpu_prog {
  gpuModule module;
  gpuFunction kernel;
};

gpurtcResult gpu_compile(gpurtcProgram prog, struct gpu_backend *nbnd);

static int gpu_update(struct nomp_backend_t *bnd, struct nomp_mem_t *m,
                      const nomp_map_direction_t op, size_t start, size_t end,
                      size_t usize) {
  if (op & NOMP_ALLOC)
    chk_gpu(gpuMalloc(&m->bptr, NOMP_MEM_BYTES(start, end, usize)));

  if (op & NOMP_TO) {
    chk_gpu(
        gpuMemcpy((char *)(m->bptr) + NOMP_MEM_OFFSET(start - m->idx0, usize),
                  (char *)(m->hptr) + NOMP_MEM_OFFSET(start, usize),
                  NOMP_MEM_BYTES(start, end, usize), gpuMemcpyHostToDevice));
  }

  if (op == NOMP_FROM) {
    chk_gpu(
        gpuMemcpy((char *)(m->hptr) + NOMP_MEM_OFFSET(start, usize),
                  (char *)(m->bptr) + NOMP_MEM_OFFSET(start - m->idx0, usize),
                  NOMP_MEM_BYTES(start, end, usize), gpuMemcpyDeviceToHost));
  } else if (op == NOMP_FREE) {
    chk_gpu(gpuFree(m->bptr));
    m->bptr = NULL;
  }

  return 0;
}

static void gpu_update_ptr(void **p, size_t *size, struct nomp_mem_t *m) {
  *p = (void *)m->bptr;
  *size = sizeof(m->bptr);
}

static int gpu_knl_build(struct nomp_backend_t *bnd, struct nomp_prog_t *prg,
                         const char *source, const char *name) {
  struct gpu_backend *nbnd = (struct gpu_backend *)bnd->bptr;

  gpurtcProgram prog;
  chk_gpurtc(gpurtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  gpurtcResult result = gpu_compile(prog, nbnd);
  if (result != GPURTC_SUCCESS) {
    size_t size;
    gpurtcGetProgramLogSize(prog, &size);
    char *log = nomp_calloc(char, size + 1);
    gpurtcGetProgramLog(prog, log);
    const char *err_str = gpurtcGetErrorString(result);
    size += strlen(err_str) + 2 + 1;

    char *msg = nomp_calloc(char, size);
    snprintf(msg, size, "%s: %s", err_str, log);
    int err_id = nomp_log(NOMP_GPU_FAILURE, NOMP_ERROR, ERR_STR_GPU_FAILURE,
                          "build", msg);
    nomp_free(&log), nomp_free(&msg);
    return err_id;
  }

  size_t code_size;
  chk_gpurtc(gpurtcGetCodeSize(prog, &code_size));
  char *code = nomp_calloc(char, code_size);
  chk_gpurtc(gpurtcGetCode(prog, code));

  struct gpu_prog *nprg = prg->bptr = nomp_calloc(struct gpu_prog, 1);
  GPU_CHECK(gpuModuleLoadData(&nprg->module, code));
  GPU_CHECK(gpuModuleGetFunction(&nprg->kernel, nprg->module, name));

  nomp_free(&code);
  chk_gpurtc(gpurtcDestroyProgram(&prog));

  return 0;
}

static int gpu_knl_run(struct nomp_backend_t *bnd, struct nomp_prog_t *prg) {
  void *vargs[NOMP_MAX_KNL_ARGS];
  struct nomp_arg_t *args = prg->args;
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

static int gpu_knl_free(struct nomp_prog_t *prg) {
  struct gpu_prog *cprg = (struct gpu_prog *)prg->bptr;

  if (cprg)
    GPU_CHECK(gpuModuleUnload(cprg->module));

  return 0;
}

static int gpu_sync(struct nomp_backend_t *bnd) {
  chk_gpu(gpuDeviceSynchronize());
  return 0;
}

static int gpu_finalize(struct nomp_backend_t *bnd) {
#ifndef __HIP_PLATFORM_HCC__
  struct gpu_backend *nbnd = (struct gpu_backend *)bnd->bptr;
  if (nbnd)
    GPU_CHECK(gpuCtxDestroy(nbnd->ctx));
#endif
  nomp_free(&bnd->bptr);
  return 0;
}

int gpu_init(struct nomp_backend_t *bnd, const int platform_id,
             const int device_id) {
  int num_devices;
  GPU_CHECK(gpuGetDeviceCount(&num_devices));
  if (device_id < 0 || device_id >= num_devices) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
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
#undef chk_err_
#undef chk_gpu
#undef chk_gpurtc

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
#undef gpu_compile

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
