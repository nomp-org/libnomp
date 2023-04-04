#include "nomp-impl.h"
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#define chk_hip_(file, line, call)                                             \
  do {                                                                         \
    hipError_t result = (call);                                                \
    if (result != hipSuccess) {                                                \
      const char *msg = hipGetErrorName(result);                               \
      return nomp_set_log(NOMP_HIP_FAILURE, NOMP_ERROR, ERR_STR_HIP_FAILURE,   \
                          "operation", msg);                                   \
    }                                                                          \
  } while (0)

#define chk_hip(call) chk_hip_(__FILE__, __LINE__, call)

#define chk_hiprtc_(file, line, call)                                          \
  do {                                                                         \
    hiprtcResult result = (call);                                              \
    if (result != HIPRTC_SUCCESS) {                                            \
      const char *msg = hiprtcGetErrorString(result);                          \
      return nomp_set_log(NOMP_HIP_FAILURE, NOMP_ERROR, ERR_STR_HIP_FAILURE,   \
                          "runtime compilation", msg);                         \
    }                                                                          \
  } while (0)
#define chk_hiprtc(call) chk_hiprtc_(__FILE__, __LINE__, call)

const char *ERR_STR_HIP_FAILURE = "HIP %s failed: %s.";

struct hip_backend {
  int device_id;
  struct hipDeviceProp_t prop;
  hipCtx_t ctx;
};

struct hip_prog {
  hipModule_t module;
  hipFunction_t kernel;
};

static int hip_update(struct backend *bnd, struct mem *m, const int op) {
  if (op & NOMP_ALLOC)
    chk_hip(hipMalloc(&m->bptr, (m->idx1 - m->idx0) * m->usize));

  if (op & NOMP_TO) {
    chk_hip(hipMemcpy(m->bptr, (char *)m->hptr + m->usize * m->idx0,
                      (m->idx1 - m->idx0) * m->usize, hipMemcpyHostToDevice));
  }

  if (op == NOMP_FROM) {
    chk_hip(hipMemcpy((char *)m->hptr + m->usize * m->idx0, m->bptr,
                      (m->idx1 - m->idx0) * m->usize, hipMemcpyDeviceToHost));
  } else if (op == NOMP_FREE) {
    chk_hip(hipFree(m->bptr));
    m->bptr = NULL;
  }

  return 0;
}

static void hip_update_ptr(void **p, size_t *size, struct mem *m) {
  *p = (void *)m->bptr;
  *size = sizeof(m->bptr);
}

static int hip_knl_build(struct backend *bnd, struct prog *prg,
                         const char *source, const char *name) {
  hiprtcProgram prog;
  chk_hiprtc(hiprtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  struct hip_backend *hbnd = (struct hip_backend *)bnd->bptr;

  const char *opts[1] = {NULL};
  hiprtcResult result = hiprtcCompileProgram(prog, 0, opts);
  if (result != HIPRTC_SUCCESS) {
    size_t log_size;
    hiprtcGetProgramLogSize(prog, &log_size);
    char *log = nomp_calloc(char, log_size);
    hiprtcGetProgramLog(prog, log);
    const char *err_str = hiprtcGetErrorString(result);
    size_t msg_size = log_size + strlen(err_str) + 3;
    char *msg = nomp_calloc(char, msg_size);
    snprintf(msg, msg_size, "%s: %s", err_str, log);
    int err_id = nomp_set_log(NOMP_HIP_FAILURE, NOMP_ERROR, ERR_STR_HIP_FAILURE,
                              "build", msg);
    nomp_free(log), nomp_free(msg);
    return err_id;
  }

  size_t code_size;
  chk_hiprtc(hiprtcGetCodeSize(prog, &code_size));

  char *code = nomp_calloc(char, code_size);
  chk_hiprtc(hiprtcGetCode(prog, code));

  chk_hiprtc(hiprtcDestroyProgram(&prog));

  struct hip_prog *hprg = nomp_calloc(struct hip_prog, 1);
  prg->bptr = hprg;
  chk_hip(hipModuleLoadData(&hprg->module, code));

  nomp_free(code);

  chk_hip(hipModuleGetFunction(&hprg->kernel, hprg->module, name));

  return 0;
}

static int hip_knl_run(struct backend *bnd, struct prog *prg, va_list args) {
  const int ndim = prg->ndim, nargs = prg->nargs;
  const size_t *global = prg->global, *local = prg->local;

  struct mem *m;
  void *vargs[MAX_KNL_ARGS];
  for (int i = 0; i < nargs; i++) {
    const char *var = va_arg(args, const char *);
    int type = va_arg(args, int);
    size_t size = va_arg(args, size_t);
    void *p = va_arg(args, void *);
    switch (type) {
    case NOMP_INT:
    case NOMP_FLOAT:
      break;
    case NOMP_PTR:
      m = mem_if_mapped(p);
      if (m == NULL) {
        return nomp_set_log(NOMP_USER_MAP_PTR_IS_INVALID, NOMP_ERROR,
                            ERR_STR_USER_MAP_PTR_IS_INVALID, p);
      }
      p = &m->bptr;
      break;
    default:
      return nomp_set_log(NOMP_USER_KNL_ARG_TYPE_IS_INVALID, NOMP_ERROR,
                          "Invalid libnomp kernel argument type %d.", type);
      break;
    }
    vargs[i] = p;
  }

  struct hip_prog *cprg = (struct hip_prog *)prg->bptr;
  chk_hip(hipModuleLaunchKernel(cprg->kernel, global[0], global[1], global[2],
                                local[0], local[1], local[2], 0, NULL, vargs,
                                NULL));
  return 0;
}

static int hip_knl_free(struct prog *prg) {
  struct hip_prog *cprg = (struct hip_prog *)prg->bptr;
  chk_hip(hipModuleUnload(cprg->module));
  return 0;
}

static int hip_sync(struct backend *bnd) {
  chk_hip(hipDeviceSynchronize());
  return 0;
}

static int hip_finalize(struct backend *bnd) {
#ifdef __HIP_PLATFORM_NVIDIA__
  struct hip_backend *hbnd = (struct hip_backend *)bnd->bptr;
  chk_hip(hipCtxDestroy(hbnd->ctx));
#endif
  return 0;
}

int hip_init(struct backend *bnd, const int platform_id, const int device_id) {
  int num_devices;
  chk_hip(hipGetDeviceCount(&num_devices));
  if (device_id < 0 || device_id >= num_devices) {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }
  chk_hip(hipSetDevice(device_id));

  struct hip_backend *hbnd = nomp_calloc(struct hip_backend, 1);
  hbnd->device_id = device_id;
  chk_hip(hipGetDeviceProperties(&hbnd->prop, device_id));
#ifdef __HIP_PLATFORM_NVIDIA__
  chk_hip(hipInit(0));
  chk_hip(hipCtxCreate(&hbnd->ctx, 0, hbnd->device_id));
#endif
  bnd->bptr = (void *)hbnd;

  bnd->update = hip_update;
  bnd->knl_build = hip_knl_build;
  bnd->knl_run = hip_knl_run;
  bnd->knl_free = hip_knl_free;
  bnd->sync = hip_sync;
  bnd->finalize = hip_finalize;

  return 0;
}
