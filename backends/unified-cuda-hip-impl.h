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
  int device;
  backendDeviceProp_t prop;
};

#define backend_prog_t TOKEN_PASTE(DRIVER, _prog_t)
struct backend_prog_t {
  backendModule module;
  backendFunction kernel;
};

static backendrtcResult backend_compile(backendrtcProgram prog) {
  return backendrtcCompileProgram(prog, 0, NULL);
}

static int backend_update(nomp_backend_t *NOMP_UNUSED(bnd), nomp_mem_t *m,
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

static int backend_knl_build(nomp_backend_t *NOMP_UNUSED(bnd), nomp_prog_t *prg,
                             const char *source, const char *name) {
  backendrtcProgram prog;
  check_rtc(backendrtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  backendrtcResult result = backend_compile(prog);
  if (result == RTC_SUCCESS)
    goto get_code;

  const char *err = backendrtcGetErrorString(result);

  size_t size;
  backendrtcGetProgramLogSize(prog, &size);
  char *log = nomp_calloc(char, size + 1);
  backendrtcGetProgramLog(prog, log);

  size += strlen(err) + 2 + 1;
  char *msg = nomp_calloc(char, size);
  snprintf(msg, size, "%s: %s", err, log);
  int ret = nomp_log(NOMP_BACKEND_FAILURE, NOMP_ERROR, ERR_STR_BACKEND_FAILURE,
                     "build", msg);
  nomp_free(&msg), nomp_free(&log);
  return ret;

get_code:
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

static int backend_knl_run(nomp_backend_t *NOMP_UNUSED(bnd), nomp_prog_t *prg) {
  nomp_arg_t *args = prg->args;
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

static int backend_knl_free(nomp_prog_t *prg) {
  struct backend_prog_t *bprg = (struct backend_prog_t *)prg->bptr;
  if (bprg)
    check_runtime(backendModuleUnload(bprg->module));
  return 0;
}

static int backend_sync(nomp_backend_t *NOMP_UNUSED(bnd)) {
  check_driver(backendDeviceSynchronize());
  return 0;
}

static int backend_finalize(nomp_backend_t *bnd) {
  nomp_free(&bnd->bptr);
  return 0;
}

static int backend_device_query(nomp_backend_t *bnd, int device) {
  backendDeviceProp_t prop;
  check_driver(backendGetDeviceProperties(&prop, device));

#define set_string(KEY, VAL)                                                   \
  {                                                                            \
    PyObject *obj = PyUnicode_FromString(VAL);                                 \
    PyDict_SetItemString(bnd->py_context, KEY, obj);                           \
    Py_XDECREF(obj);                                                           \
  }

#define set_int(KEY, VAL)                                                      \
  {                                                                            \
    PyObject *obj = PyLong_FromSize_t(VAL);                                    \
    PyDict_SetItemString(bnd->py_context, KEY, obj);                           \
    Py_XDECREF(obj);                                                           \
  }

  set_string("device::name", prop.name);

#if defined(NOMP_HIP)
  set_string("device::vendor", "AMD");
#elif defined(NOMP_CUDA)
  set_string("device::vendor", "NVIDIA");
#endif

  int driver_version;
  check_driver(backendDriverGetVersion(&driver_version));
  set_int("device::driver", driver_version);

  set_string("device::type", "gpu");

  set_int("device::max_threads_per_block", prop.maxThreadsPerBlock);

#undef set_int
#undef set_string

  return 0;
}

#define backend_init TOKEN_PASTE(DRIVER, _init)
int backend_init(nomp_backend_t *const backend, const int NOMP_UNUSED(platform),
                 const int device) {
  int num_devices;
  check_driver(backendGetDeviceCount(&num_devices));
  if (device < 0 || device >= num_devices) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    ERR_STR_USER_DEVICE_IS_INVALID, device);
  }

  check_driver(backendSetDevice(device));
  check_driver(backendFree(0));

  nomp_check(backend_device_query(backend, device));

  struct backend_t *bptr = nomp_calloc(struct backend_t, 1);
  bptr->device = device;
  check_driver(backendGetDeviceProperties(&bptr->prop, device));

  backend->bptr = (void *)bptr;
  backend->update = backend_update;
  backend->knl_build = backend_knl_build;
  backend->knl_run = backend_knl_run;
  backend->knl_free = backend_knl_free;
  backend->sync = backend_sync;
  backend->finalize = backend_finalize;

  return 0;
}

#undef backend_init
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

#undef backendrtcDestroyProgram
#undef backendrtcGetProgramLog
#undef backendrtcGetProgramLogSize
#undef backendrtcCompileProgram
#undef backendrtcCreateProgram
#undef backendrtcProgram
#undef backendrtcGetErrorString
#undef backendrtcResult

#undef backendDriverGetVersion
#undef backendGetDeviceProperties
#undef backendSetDevice
#undef backendGetDeviceCount
#undef backendDeviceSynchronize
#undef backendMemcpyDeviceToHost
#undef backendMemcpyHostToDevice
#undef backendFree
#undef backendMemcpy
#undef backendMalloc
#undef backendGetErrorName
#undef backendSuccess
#undef backendError_t

#undef TOKEN_PASTE
#undef TOKEN_PASTE_
