#include "nomp-impl.h"

#define TOKEN_PASTE_(a, b) a##b
#define TOKEN_PASTE(a, b) TOKEN_PASTE_(a, b)

#define backendError_t TOKEN_PASTE(DRIVER, Error_t)
#define backendSuccess TOKEN_PASTE(DRIVER, Success)
#define backendGetErrorName TOKEN_PASTE(DRIVER, GetErrorName)
#define backendMalloc TOKEN_PASTE(DRIVER, Malloc)
#define backendMemcpy TOKEN_PASTE(DRIVER, Memcpy)
#define backendFree TOKEN_PASTE(DRIVER, Free)
#define backendMemcpyHostToDevice TOKEN_PASTE(DRIVER, MemcpyHostToDevice)
#define backendMemcpyDeviceToHost TOKEN_PASTE(DRIVER, MemcpyDeviceToHost)
#define backendDeviceSynchronize TOKEN_PASTE(DRIVER, DeviceSynchronize)
#define backendGetDeviceCount TOKEN_PASTE(DRIVER, GetDeviceCount)
#define backendSetDevice TOKEN_PASTE(DRIVER, SetDevice)
#define backendGetDeviceProperties TOKEN_PASTE(DRIVER, GetDeviceProperties)
#define backendDriverGetVersion TOKEN_PASTE(DRIVER, DriverGetVersion)

#define backendrtcResult TOKEN_PASTE(RUNTIME_COMPILATION, Result)
#define backendrtcGetErrorString                                               \
  TOKEN_PASTE(RUNTIME_COMPILATION, GetErrorString)
#define backendrtcProgram TOKEN_PASTE(RUNTIME_COMPILATION, Program)
#define backendrtcCreateProgram TOKEN_PASTE(RUNTIME_COMPILATION, CreateProgram)
#define backendrtcCompileProgram                                               \
  TOKEN_PASTE(RUNTIME_COMPILATION, CompileProgram)
#define backendrtcGetProgramLogSize                                            \
  TOKEN_PASTE(RUNTIME_COMPILATION, GetProgramLogSize)
#define backendrtcGetProgramLog TOKEN_PASTE(RUNTIME_COMPILATION, GetProgramLog)
#define backendrtcDestroyProgram                                               \
  TOKEN_PASTE(RUNTIME_COMPILATION, DestroyProgram)

#define backendInit TOKEN_PASTE(RUNTIME, Init)
#define backendCtxCreate TOKEN_PASTE(RUNTIME, CtxCreate)
#define backendCtxDestroy TOKEN_PASTE(RUNTIME, CtxDestroy)
#define backendModuleLoadData TOKEN_PASTE(RUNTIME, ModuleLoadData)
#define backendModuleGetFunction TOKEN_PASTE(RUNTIME, ModuleGetFunction)
#define backendModuleUnload TOKEN_PASTE(RUNTIME, ModuleUnload)

#define check_error(CALL, ERR_T, SUCCES, GET_ERR, OP)                          \
  {                                                                            \
    ERR_T result = (CALL);                                                     \
    if (result != SUCCES) {                                                    \
      const char *msg = GET_ERR(result);                                       \
      return nomp_log(NOMP_BACKEND_FAILURE, NOMP_ERROR,                        \
                      ERR_STR_BACKEND_FAILURE, OP, msg);                       \
    }                                                                          \
  }

#define check_driver(call)                                                     \
  check_error(call, backendError_t, backendSuccess, backendGetErrorName,       \
              "driver");

#define check_rtc(call)                                                        \
  check_error(call, backendrtcResult, RTC_SUCCESS, backendrtcGetErrorString,   \
              "runtime")

#define backend_t TOKEN_PASTE(DRIVER, _backend_t)
struct backend_t {
  int device_id;
  backendDeviceProp_t prop;
};

#define backend_prog_t TOKEN_PASTE(DRIVER, _prog_t)
struct backend_prog_t {
  backendModule module;
  backendFunction kernel;
};

#define backend_compile TOKEN_PASTE(DRIVER, _compile)
static backendrtcResult backend_compile(backendrtcProgram prog,
                                        struct backend_t *bnd) {
  char arch[NOMP_MAX_BUFFER_SIZE];
  snprintf(arch, NOMP_MAX_BUFFER_SIZE, "-arch=compute_%d%d", bnd->prop.major,
           bnd->prop.minor);
  const char *opts[2] = {arch, NULL};
  return backendrtcCompileProgram(prog, 1, opts);
}

#define backend_update TOKEN_PASTE(DRIVER, _update)
static int backend_update(struct nomp_backend_t *bnd, struct nomp_mem_t *m,
                          const nomp_map_direction_t op, size_t start,
                          size_t end, size_t usize) {
  if (op & NOMP_ALLOC)
    check_driver(backendMalloc(&m->bptr, NOMP_MEM_BYTES(start, end, usize)));

  if (op & NOMP_TO) {
    check_driver(backendMemcpy(
        (char *)(m->bptr) + NOMP_MEM_OFFSET(start - m->idx0, usize),
        (char *)(m->hptr) + NOMP_MEM_OFFSET(start, usize),
        NOMP_MEM_BYTES(start, end, usize), backendMemcpyHostToDevice));
  }

  if (op == NOMP_FROM) {
    check_driver(backendMemcpy(
        (char *)(m->hptr) + NOMP_MEM_OFFSET(start, usize),
        (char *)(m->bptr) + NOMP_MEM_OFFSET(start - m->idx0, usize),
        NOMP_MEM_BYTES(start, end, usize), backendMemcpyDeviceToHost));
  } else if (op == NOMP_FREE) {
    check_driver(backendFree(m->bptr));
    m->bptr = NULL;
  }

  return 0;
}

#define backend_update_ptr TOKEN_PASTE(DRIVER, _update_ptr)
static void backend_update_ptr(void **p, size_t *size, struct nomp_mem_t *m) {
  *p = (void *)m->bptr, *size = sizeof(m->bptr);
}

#define backend_knl_build TOKEN_PASTE(DRIVER, _knl_build)
static int backend_knl_build(struct nomp_backend_t *bnd,
                             struct nomp_prog_t *prg, const char *source,
                             const char *name) {
  struct backend_t *backend = (struct backend_t *)bnd->bptr;

  backendrtcProgram prog;
  check_rtc(backendrtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  backendrtcResult result = backend_compile(prog, backend);
  if (result != RTC_SUCCESS) {
    const char *err = backendrtcGetErrorString(result);

    size_t size;
    backendrtcGetProgramLogSize(prog, &size);
    char *log = nomp_calloc(char, size + 1);
    backendrtcGetProgramLog(prog, log);

    size += strlen(err) + 2 + 1;
    char *msg = nomp_calloc(char, size);
    snprintf(msg, size, "%s: %s", err, log);
    int ret = nomp_log(NOMP_BACKEND_FAILURE, NOMP_ERROR,
                       ERR_STR_BACKEND_FAILURE, "build", msg);
    nomp_free(&msg), nomp_free(&log);
    return ret;
  }

  size_t size;
  check_rtc(backendrtcGetCodeSize(prog, &size));
  char *code = nomp_calloc(char, size + 1);
  check_rtc(backendrtcGetCode(prog, code));
  check_rtc(backendrtcDestroyProgram(&prog));

  struct backend_prog_t *gprg = nomp_calloc(struct backend_prog_t, 1);
  check_runtime(backendModuleLoadData(&gprg->module, code));
  nomp_free(&code);
  check_runtime(backendModuleGetFunction(&gprg->kernel, gprg->module, name));
  prg->bptr = (void *)gprg;

  return 0;
}

#define backend_knl_run TOKEN_PASTE(DRIVER, _knl_run)
static int backend_knl_run(struct nomp_backend_t *bnd,
                           struct nomp_prog_t *prg) {
  struct nomp_arg_t *args = prg->args;
  void *vargs[NOMP_MAX_KERNEL_ARGS_SIZE];
  for (unsigned i = 0; i < prg->nargs; i++) {
    if (args[i].type == NOMP_PTR)
      vargs[i] = &args[i].ptr;
    else
      vargs[i] = args[i].ptr;
  }

  const size_t *global = prg->global, *local = prg->local;
  struct backend_prog_t *bprg = (struct backend_prog_t *)prg->bptr;
  check_runtime(backendModuleLaunchKernel(bprg->kernel, global[0], global[1],
                                          global[2], local[0], local[1],
                                          local[2], 0, NULL, vargs, NULL));

  return 0;
}

#define backend_knl_free TOKEN_PASTE(DRIVER, _knl_free)
static int backend_knl_free(struct nomp_prog_t *prg) {
  struct backend_prog_t *bprg = (struct backend_prog_t *)prg->bptr;
  if (bprg)
    check_runtime(backendModuleUnload(bprg->module));
  return 0;
}

#define backend_sync TOKEN_PASTE(DRIVER, _sync)
static int backend_sync(struct nomp_backend_t *bnd) {
  check_driver(backendDeviceSynchronize());
  return 0;
}

#define backend_finalize TOKEN_PASTE(DRIVER, _finalize)
static int backend_finalize(struct nomp_backend_t *bnd) {
  nomp_free(&bnd->bptr);
  return 0;
}

#define backend_device_query TOKEN_PASTE(DRIVER, _device_query)
static int backend_device_query(struct nomp_backend_t *bnd, int device_id) {
  backendDeviceProp_t prop;
  check_driver(backendGetDeviceProperties(&prop, device_id));

#define set_string_aux(KEY, VAL)                                               \
  {                                                                            \
    PyObject *obj = PyUnicode_FromString(VAL);                                 \
    PyDict_SetItemString(bnd->py_context, KEY, obj);                           \
    Py_XDECREF(obj);                                                           \
  }

  set_string_aux("device::name", prop.name);

#if defined(NOMP_HIP)
  set_string_aux("device::vendor", "AMD");
#elif defined(NOMP_CUDA)
  set_string_aux("device::vendor", "NVIDIA");
#endif

#define set_int_aux(KEY, VAL)                                                  \
  {                                                                            \
    PyObject *obj = PyLong_FromSize_t(VAL);                                    \
    PyDict_SetItemString(bnd->py_context, KEY, obj);                           \
    Py_XDECREF(obj);                                                           \
  }

  int driver_version;
  check_driver(backendDriverGetVersion(&driver_version));
  set_int_aux("device::driver", driver_version);

  set_string_aux("device::type", "gpu");

#undef set_int_aux
#undef set_string_aux

  return 0;
}

#define backend_init TOKEN_PASTE(DRIVER, _init)
int backend_init(struct nomp_backend_t *bnd, const int platform_id,
                 const int device_id) {
  int num_devices;
  check_driver(backendGetDeviceCount(&num_devices));
  if (device_id < 0 || device_id >= num_devices) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }

  check_driver(backendSetDevice(device_id));
  check_driver(backendFree(0));

  nomp_check(backend_device_query(bnd, device_id));

  struct backend_t *backend = nomp_calloc(struct backend_t, 1);
  backend->device_id = device_id;
  check_driver(backendGetDeviceProperties(&backend->prop, device_id));

  bnd->bptr = (void *)backend;
  bnd->update = backend_update;
  bnd->knl_build = backend_knl_build;
  bnd->knl_run = backend_knl_run;
  bnd->knl_free = backend_knl_free;
  bnd->sync = backend_sync;
  bnd->finalize = backend_finalize;

  return 0;
}

#undef backend_init
#undef backend_finalize
#undef backend_sync
#undef backend_knl_free
#undef backend_knl_run
#undef backend_knl_build
#undef backend_update_ptr
#undef backend_update
#undef backend_compile

#undef backend_prog_t
#undef backend_t

#undef check_rtc
#undef check_driver
#undef check_error

#undef backendModuleUnload
#undef backendModuleGetFunction
#undef backendModuleLoadData
#undef backendCtxDestroy
#undef backendCtxCreate
#undef backendInit

#undef backendError_t
#undef backendSuccess
#undef backendGetErrorName
#undef backendMalloc
#undef backendMemcpy
#undef backendFree
#undef backendMemcpyHostToDevice
#undef backendMemcpyDeviceToHost
#undef backendDeviceSynchronize
#undef backendGetDeviceCount
#undef backendSetDevice
#undef backendGetDeviceProperties
#undef backendDriverGetVersion

#undef backendrtcResult
#undef backendrtcGetErrorString
#undef backendrtcProgram
#undef backendrtcCreateProgram
#undef backendrtcCompileProgram
#undef backendrtcGetProgramLogSize
#undef backendrtcGetProgramLog
#undef backendrtcDestroyProgram

#undef TOKEN_PASTE
#undef TOKEN_PASTE_
