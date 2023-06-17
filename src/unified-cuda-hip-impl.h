#include "nomp-impl.h"

#define TOKEN_PASTE_(a, b) a##b
#define TOKEN_PASTE(a, b) TOKEN_PASTE_(a, b)

#define gpuError_t TOKEN_PASTE(BACKEND, Error_t)
#define gpuSuccess TOKEN_PASTE(BACKEND, Success)
#define gpuGetErrorName TOKEN_PASTE(BACKEND, GetErrorName)
#define gpuMalloc TOKEN_PASTE(BACKEND, Malloc)
#define gpuMemcpy TOKEN_PASTE(BACKEND, Memcpy)
#define gpuFree TOKEN_PASTE(BACKEND, Free)
#define gpuMemcpyHostToDevice TOKEN_PASTE(BACKEND, MemcpyHostToDevice)
#define gpuMemcpyDeviceToHost TOKEN_PASTE(BACKEND, MemcpyDeviceToHost)
#define gpuDeviceSynchronize TOKEN_PASTE(BACKEND, DeviceSynchronize)
#define gpuGetDeviceCount TOKEN_PASTE(BACKEND, GetDeviceCount)
#define gpuSetDevice TOKEN_PASTE(BACKEND, SetDevice)
#define gpuGetDeviceProperties TOKEN_PASTE(BACKEND, GetDeviceProperties)

#define gpurtcResult TOKEN_PASTE(RUNTIME, Result)
#define gpurtcGetErrorString TOKEN_PASTE(RUNTIME, GetErrorString)
#define gpurtcProgram TOKEN_PASTE(RUNTIME, Program)
#define gpurtcCreateProgram TOKEN_PASTE(RUNTIME, CreateProgram)
#define gpurtcCompileProgram TOKEN_PASTE(RUNTIME, CompileProgram)
#define gpurtcGetProgramLogSize TOKEN_PASTE(RUNTIME, GetProgramLogSize)
#define gpurtcGetProgramLog TOKEN_PASTE(RUNTIME, GetProgramLog)
#define gpurtcDestroyProgram TOKEN_PASTE(RUNTIME, DestroyProgram)

#define chk_err(CALL, ERR_T, SUCCES, GETERR, OP)                               \
  {                                                                            \
    ERR_T result = (CALL);                                                     \
    if (result != SUCCES) {                                                    \
      const char *msg = GETERR(result);                                        \
      return nomp_log(NOMP_BACKEND_FAILURE, NOMP_ERROR,                        \
                      ERR_STR_BACKEND_FAILURE, OP, msg);                       \
    }                                                                          \
  }

#define chk_gpu(call)                                                          \
  chk_err(call, gpuError_t, gpuSuccess, gpuGetErrorName, "operation");

#define chk_gpu_rt(call)                                                       \
  chk_err(call, gpurtcResult, GPURTC_SUCCESS, gpurtcGetErrorString, "runtime")

#define gpu_backend_t TOKEN_PASTE(BACKEND, _backend_t)
struct gpu_backend_t {
  int device_id;
  gpuDeviceProp_t prop;
};

#define gpu_prog_t TOKEN_PASTE(BACKEND, _prog_t)
struct gpu_prog_t {
  gpuModule module;
  gpuFunction kernel;
};

#define gpu_compile TOKEN_PASTE(BACKEND, _compile)
static gpurtcResult gpu_compile(gpurtcProgram prog, struct gpu_backend_t *bnd) {
  char arch[NOMP_MAX_BUFSIZ];
  snprintf(arch, NOMP_MAX_BUFSIZ, "-arch=compute_%d%d", bnd->prop.major,
           bnd->prop.minor);
  const char *opts[2] = {arch, NULL};
  return gpurtcCompileProgram(prog, 1, opts);
}

#define gpu_update TOKEN_PASTE(BACKEND, _update)
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

#define gpu_update_ptr TOKEN_PASTE(BACKEND, _update_ptr)
static void gpu_update_ptr(void **p, size_t *size, struct nomp_mem_t *m) {
  *p = (void *)m->bptr, *size = sizeof(m->bptr);
}

#define gpu_knl_build TOKEN_PASTE(BACKEND, _knl_build)
static int gpu_knl_build(struct nomp_backend_t *bnd, struct nomp_prog_t *prg,
                         const char *source, const char *name) {
  struct gpu_backend_t *gbnd = (struct gpu_backend_t *)bnd->bptr;

  gpurtcProgram prog;
  chk_gpu_rt(gpurtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  gpurtcResult result = gpu_compile(prog, gbnd);
  if (result != GPURTC_SUCCESS) {
    const char *err = gpurtcGetErrorString(result);

    size_t size;
    gpurtcGetProgramLogSize(prog, &size);
    char *log = nomp_calloc(char, size + 1);
    gpurtcGetProgramLog(prog, log);

    size += strlen(err) + 2 + 1;
    char *msg = nomp_calloc(char, size);
    snprintf(msg, size, "%s: %s", err, log);
    int ret = nomp_log(NOMP_BACKEND_FAILURE, NOMP_ERROR,
                       ERR_STR_BACKEND_FAILURE, "build", msg);
    nomp_free(&msg), nomp_free(&log);
    return ret;
  }

  size_t size;
  chk_gpu_rt(gpurtcGetCodeSize(prog, &size));
  char *code = nomp_calloc(char, size + 1);
  chk_gpu_rt(gpurtcGetCode(prog, code));
  chk_gpu_rt(gpurtcDestroyProgram(&prog));

  struct gpu_prog_t *gprg = nomp_calloc(struct gpu_prog_t, 1);
  check(gpuModuleLoadData(&gprg->module, code));
  nomp_free(&code);
  check(gpuModuleGetFunction(&gprg->kernel, gprg->module, name));
  prg->bptr = (void *)gprg;

  return 0;
}

#define gpu_knl_run TOKEN_PASTE(BACKEND, _knl_run)
static int gpu_knl_run(struct nomp_backend_t *bnd, struct nomp_prog_t *prg) {
  struct nomp_arg_t *args = prg->args;
  void *vargs[NOMP_MAX_KNL_ARGS];
  for (int i = 0; i < prg->nargs; i++) {
    if (args[i].type == NOMP_PTR)
      vargs[i] = &args[i].ptr;
    else
      vargs[i] = args[i].ptr;
  }

  const size_t *global = prg->global, *local = prg->local;
  struct gpu_prog_t *gprg = (struct gpu_prog_t *)prg->bptr;
  check(gpuModuleLaunchKernel(gprg->kernel, global[0], global[1], global[2],
                              local[0], local[1], local[2], 0, NULL, vargs,
                              NULL));

  return 0;
}

#define gpu_knl_free TOKEN_PASTE(BACKEND, _knl_free)
static int gpu_knl_free(struct nomp_prog_t *prg) {
  struct gpu_prog_t *gprg = (struct gpu_prog_t *)prg->bptr;
  if (gprg)
    check(gpuModuleUnload(gprg->module));

  return 0;
}

#define gpu_sync TOKEN_PASTE(BACKEND, _sync)
static int gpu_sync(struct nomp_backend_t *bnd) {
  chk_gpu(gpuDeviceSynchronize());
  return 0;
}

#define gpu_finalize TOKEN_PASTE(BACKEND, _finalize)
static int gpu_finalize(struct nomp_backend_t *bnd) {
  nomp_free(&bnd->bptr);
  return 0;
}

#define gpu_init TOKEN_PASTE(BACKEND, _init)
int gpu_init(struct nomp_backend_t *bnd, const int platform_id,
             const int device_id) {
  int num_devices;
  check(gpuGetDeviceCount(&num_devices));
  if (device_id < 0 || device_id >= num_devices) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }

  chk_gpu(gpuSetDevice(device_id));
  check(gpuFree(0));

  struct gpu_backend_t *gbnd = nomp_calloc(struct gpu_backend_t, 1);
  gbnd->device_id = device_id;
  chk_gpu(gpuGetDeviceProperties(&gbnd->prop, device_id));

  bnd->bptr = (void *)gbnd;
  bnd->update = gpu_update;
  bnd->knl_build = gpu_knl_build;
  bnd->knl_run = gpu_knl_run;
  bnd->knl_free = gpu_knl_free;
  bnd->sync = gpu_sync;
  bnd->finalize = gpu_finalize;

  return 0;
}

#undef gpu_init
#undef gpu_finalize
#undef gpu_sync
#undef gpu_knl_free
#undef gpu_knl_run
#undef gpu_knl_build
#undef gpu_update_ptr
#undef gpu_update
#undef gpu_compile

#undef gpu_prog_t
#undef gpu_backend_t

#undef chk_gpu_rt
#undef chk_gpu
#undef chk_err

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

#undef TOKEN_PASTE
#undef TOKEN_PASTE_
