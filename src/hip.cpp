extern "C" {
#include "nomp-impl.h"
}
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#define NARGS_MAX 64

#define chk_hip_(file, line, x)                                                \
  do {                                                                         \
    if (x != hipSuccess) {                                                     \
      const char *msg = hipGetErrorName(x);                                    \
      return set_log(NOMP_HIP_FAILURE, NOMP_ERROR, ERR_STR_HIP_FAILURE,        \
                     "operation", msg);                                        \
    }                                                                          \
  } while (0)

#define chk_hip(x) chk_hip_(__FILE__, __LINE__, x)

#define chk_hiprtc_(file, line, x)                                             \
  do {                                                                         \
    if (x != HIPRTC_SUCCESS) {                                                 \
      const char *msg = hiprtcGetErrorString(x);                               \
      return set_log(NOMP_HIP_FAILURE, NOMP_ERROR, ERR_STR_HIP_FAILURE,        \
                     "runtime compilation", msg);                              \
    }                                                                          \
  } while (0)
#define chk_hiprtc(x) chk_hiprtc_(__FILE__, __LINE__, x)

const char *ERR_STR_HIP_FAILURE = "HIP %s failed: %s.";

struct hip_backend {
  int device_id;
  struct hipDeviceProp_t prop;
};

struct hip_prog {
  hipModule_t module;
  hipFunction_t kernel;
};

static int hip_update(struct backend *bnd, struct mem *m, const int op) {
  hipError_t err;
  if (op & NOMP_ALLOC) {
    err = hipMalloc(&m->bptr, (m->idx1 - m->idx0) * m->usize);
    chk_hip(err);
  }

  if (op & NOMP_TO) {
    err = hipMemcpy(m->bptr, (char *)m->hptr + m->usize * m->idx0,
                    (m->idx1 - m->idx0) * m->usize, hipMemcpyHostToDevice);
    chk_hip(err);
  }

  if (op == NOMP_FROM) {
    err = hipMemcpy((char *)m->hptr + m->usize * m->idx0, m->bptr,
                    (m->idx1 - m->idx0) * m->usize, hipMemcpyDeviceToHost);
    chk_hip(err);
  } else if (op == NOMP_FREE) {
    err = hipFree(m->bptr);
    chk_hip(err);
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
  hiprtcResult hiprtc_err =
      hiprtcCreateProgram(&prog, source, NULL, 0, NULL, NULL);
  chk_hiprtc(hiprtc_err);

  struct hip_backend *hbnd = (struct hip_backend *)bnd->bptr;
  char arch[NOMP_BUFSIZ];
  snprintf(arch, NOMP_BUFSIZ, "-arch=compute_%d%d", hbnd->prop.major,
           hbnd->prop.minor);

  const char *opts[1] = {arch};
  hiprtc_err = hiprtcCompileProgram(prog, 1, opts);
  if (hiprtc_err != HIPRTC_SUCCESS) {
    size_t log_size;
    hiprtcGetProgramLogSize(prog, &log_size);
    char *log = tcalloc(char, log_size);
    hiprtcGetProgramLog(prog, log);
    const char *err_str = hiprtcGetErrorString(hiprtc_err);
    size_t msg_size = log_size + strlen(err_str) + 2;
    char *msg = tcalloc(char, msg_size);
    snprintf(msg, msg_size, "%s: %s", err_str, log);
    int err_id = set_log(NOMP_CUDA_FAILURE, NOMP_ERROR, ERR_STR_CUDA_FAILURE,
                         "build", msg);
    tfree(log), tfree(msg);
    return err_id;
  }

  size_t code_size;
  hiprtc_err = hiprtcGetCodeSize(prog, &code_size);
  chk_hiprtc(hiprtc_err);

  char *code = tcalloc(char, code_size);
  hiprtc_err = hiprtcGetCode(prog, code);
  chk_hiprtc(hiprtc_err);

  hiprtc_err = hiprtcDestroyProgram(&prog);
  chk_hiprtc(hiprtc_err);

  struct hip_prog *hprg = tcalloc(struct hip_prog, 1);
  prg->bptr = hprg;
  hipError_t hip_err = hipModuleLoadData(&hprg->module, code);
  chk_hip(hip_err);

  tfree(code);

  hip_err = hipModuleGetFunction(&hprg->kernel, hprg->module, name);
  chk_hip(hip_err);

  return 0;
}

static int hip_knl_run(struct backend *bnd, struct prog *prg, va_list args) {
  const int ndim = prg->ndim, nargs = prg->nargs;
  const size_t *global = prg->global, *local = prg->local;

  struct mem *m;
  void *vargs[NARGS_MAX];
  for (int i = 0; i < nargs; i++) {
    const char *var = va_arg(args, const char *);
    int type = va_arg(args, int);
    size_t size = va_arg(args, size_t);
    void *p = va_arg(args, void *);
    switch (type) {
    case NOMP_INTEGER:
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

  struct hip_prog *cprg = (struct hip_prog *)prg->bptr;
  hipError_t err =
      hipModuleLaunchKernel(cprg->kernel, global[0], global[1], global[2],
                            local[0], local[1], local[2], 0, NULL, vargs, NULL);
  return err != hipSuccess;
}

static int hip_knl_free(struct prog *prg) {
  struct hip_prog *cprg = (struct hip_prog *)prg->bptr;
  int err = hipModuleUnload(cprg->module);
  return 0;
}

static int hip_finalize(struct backend *bnd) {
  // Nothing to do
  return 0;
}

int hip_init(struct backend *bnd, const int platform_id, const int device_id) {
  int num_devices;
  hipError_t result = hipGetDeviceCount(&num_devices);
  chk_hip(result);
  if (device_id < 0 || device_id >= num_devices)
    return set_log(NOMP_USER_DEVICE_IS_INVALID, NOMP_ERROR,
                   ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  result = hipSetDevice(device_id);
  chk_hip(result);

  bnd->bptr = tcalloc(struct hip_backend, 1);
  struct hip_backend *hbnd = (struct hip_backend *)bnd->bptr;
  hbnd->device_id = device_id;
  result = hipGetDeviceProperties(&hbnd->prop, device_id);
  chk_hip(result);

  bnd->update = hip_update;
  bnd->knl_build = hip_knl_build;
  bnd->knl_run = hip_knl_run;
  bnd->knl_free = hip_knl_free;
  bnd->finalize = hip_finalize;

  return 0;
}

#undef NARGS_MAX
