#include "nomp-impl.h"
#include "nomp-jit.h"
#include <ispcrt/ispcrt.h>

// TODO: Handle errors properly in ISPC backend

#define NARGS_MAX 64

static const char *ERR_STR_ISPC_FAILURE = "ISPC %s failed with error code: %d.";

struct ispc_backend {
  ISPCRTDevice device;
  ISPCRTTaskQueue queue;
};

struct ispc_prog {
  ISPCRTModule module;
  ISPCRTKernel kernel;
};

static ISPCRTError rt_error = ISPCRT_NO_ERROR;
static char *err_message = NULL;
static void ispcrt_error(ISPCRTError err_code, const char *message) {
  rt_error = err_code;
  err_message = (char *)message;
}

#define chk_ispcrt(msg, x)                                                     \
  {                                                                            \
    if (x != ISPCRT_NO_ERROR)                                                  \
      return nomp_set_log(NOMP_ISPC_FAILURE, NOMP_ERROR, ERR_STR_ISPC_FAILURE, \
                          msg, err_message);                                   \
  }

static int ispc_update(struct backend *bnd, struct mem *m, const int op) {
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;

  if (op & NOMP_ALLOC) {
    ISPCRTNewMemoryViewFlags flags;
    flags.allocType = ISPCRT_ALLOC_TYPE_DEVICE;
    ISPCRTMemoryView view = ispcrtNewMemoryView(
        ispc->device, m->hptr, (m->idx1 - m->idx0) * m->usize, &flags);
    m->bptr = view;
  }

  if (op & NOMP_TO)
    ispcrtCopyToDevice(ispc->queue, (ISPCRTMemoryView)(m->bptr));

  if (op == NOMP_FROM) {
    ispcrtCopyToHost(ispc->queue, (ISPCRTMemoryView)(m->bptr));
  } else if (op == NOMP_FREE) {
    ispcrtRelease((ISPCRTMemoryView)(m->bptr));
    m->bptr = NULL;
  }

  // FIXME: Wrong. call set_log()
  return rt_error != ISPCRT_NO_ERROR;
}
static int ispc_knl_build(struct backend *bnd, struct prog *prg,
                          const char *source, const char *name) {
  struct ispc_backend *ispc = bnd->bptr;
  struct ispc_prog *ispc_prg = prg->bptr = nomp_calloc(struct ispc_prog, 1);

  char cwd[BUFSIZ];
  if (getcwd(cwd, BUFSIZ) == NULL) {
    return nomp_set_log(NOMP_ISPC_FAILURE, NOMP_ERROR, ERR_STR_ISPC_FAILURE,
                        "kernel build", rt_error);
  }

  char *wkdir = nomp_str_cat(3, BUFSIZ, cwd, "/", ".nomp_jit_cache");
  int ispc_id = -1;
  int err = jit_compile(&ispc_id, source, "ispc", ISPCRT_INCLUDE_DIR_FLAGS,
                        NULL, wkdir, "simple.ispc", "simple.dev.o", 1);
  int cc_id = -1;
  err = jit_compile(&cc_id, source, "/usr/bin/cc", "-fPIC -shared", NULL, wkdir,
                    "simple.dev.o", "libsimple.so", 0);

  // Create module and kernel to execute
  ISPCRTModuleOptions options = {};
  ispc_prg->module = ispcrtLoadModule(ispc->device, "simple", options);
  if (rt_error != ISPCRT_NO_ERROR) {
    ispc_prg->module = NULL;
    return nomp_set_log(NOMP_ISPC_FAILURE, NOMP_ERROR, ERR_STR_ISPC_FAILURE,
                        "module load", rt_error);
  }

  ispc_prg->kernel =
      ispcrtNewKernel(ispc->device, ispc_prg->module, "main_ispc");
  if (rt_error != ISPCRT_NO_ERROR) {
    ispc_prg->module = NULL;
    ispc_prg->kernel = NULL;
    return nomp_set_log(NOMP_ISPC_FAILURE, NOMP_ERROR, ERR_STR_ISPC_FAILURE,
                        "kernel build", rt_error);
  }
  return 0;
}

static int ispc_knl_run(struct backend *bnd, struct prog *prg, va_list args) {
  const int ndim = prg->ndim, nargs = prg->nargs;
  const size_t *global = prg->global;
  size_t num_bytes = 0;

  int i;
  struct mem *m;
  void **temp;
  void **vargs = calloc(3, sizeof(int));
  for (i = 0; i < nargs; i++) {
    const char *var = va_arg(args, const char *);
    int type = va_arg(args, int);
    size_t size = va_arg(args, size_t);
    void *p = va_arg(args, void *);
    switch (type) {
    case NOMP_INT:
    case NOMP_UINT:
    case NOMP_FLOAT:
      temp = realloc(vargs, size + sizeof(*vargs));
      vargs = temp;
      vargs[i] = calloc(1, size);
      break;
    case NOMP_PTR:
      m = mem_if_mapped(p);
      if (m == NULL)
        return nomp_set_log(NOMP_USER_MAP_PTR_IS_INVALID, NOMP_ERROR,
                            ERR_STR_USER_MAP_PTR_IS_INVALID, p);
      p = ispcrtDevicePtr((ISPCRTMemoryView)(m->bptr));
      temp = realloc(vargs, sizeof(void *) + sizeof(*vargs));
      vargs = temp;
      vargs[i] = calloc(1, (sizeof(void *)));
      break;
    default:
      return nomp_set_log(NOMP_USER_KNL_ARG_TYPE_IS_INVALID, NOMP_ERROR,
                          "Kernel argument type %d is not valid.", type);
      break;
    }
    vargs[i] = p;
  }
  for (int d = 0; d < 3; d++) {
    vargs[i + d] = calloc(1, (sizeof(int)));
    if (d < ndim)
      *((int *)(vargs[i + d])) = global[d];
    else
      *((int *)(vargs[i + d])) = 1;
  }

  ISPCRTNewMemoryViewFlags flags;
  flags.allocType = ISPCRT_ALLOC_TYPE_DEVICE;
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;
  ISPCRTMemoryView params =
      ispcrtNewMemoryView(ispc->device, vargs, num_bytes, &flags);
  ispcrtCopyToDevice(ispc->queue, params);

  // launch kernel
  struct ispc_prog *ispc_prg = (struct ispc_prog *)prg->bptr;
  ispcrtLaunch3D(ispc->queue, ispc_prg->kernel, params, 1, 1, 1);
  ispcrtSync(ispc->queue);
  chk_ispcrt("kernel run", rt_error);
  return 0;
}

static int ispc_knl_free(struct prog *prg) {
  struct ispc_prog *iprg = (struct ispc_prog *)prg->bptr;
  ispcrtRelease(iprg->kernel);
  chk_ispcrt("kernel release", rt_error);
  ispcrtRelease(iprg->module);
  chk_ispcrt("module release", rt_error);
  return 0;
}

static int ispc_finalize(struct backend *bnd) {
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;
  ispcrtRelease(ispc->device);
  chk_ispcrt("device release", rt_error);
  ispcrtRelease(ispc->queue);
  chk_ispcrt("queue release", rt_error);
  nomp_free(bnd->bptr), bnd->bptr = NULL;
  return 0;
}

static int nomp_to_ispc_device[3] = {
    ISPCRT_DEVICE_TYPE_CPU, ISPCRT_DEVICE_TYPE_GPU, ISPCRT_DEVICE_TYPE_AUTO};

int ispc_init(struct backend *bnd, const int platform_type,
              const int device_id) {
  ispcrtSetErrorFunc(ispcrt_error);
  if (platform_type < 0 | platform_type >= 3) {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "Platform type %d provided to libnomp is not valid.",
                        platform_type);
  }
  uint32_t num_devices =
      ispcrtGetDeviceCount(nomp_to_ispc_device[platform_type]);
  chk_ispcrt("get device count", rt_error);
  if (device_id < 0 || device_id >= num_devices)
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  ISPCRTDevice device = ispcrtGetDevice(platform_type, device_id);
  chk_ispcrt("device get", rt_error);

  struct ispc_backend *ispc = bnd->bptr = nomp_calloc(struct ispc_backend, 1);
  ispc->device = device;
  ispc->queue = ispcrtNewTaskQueue(device);
  chk_ispcrt("context create", rt_error);

  bnd->update = ispc_update;
  bnd->knl_build = ispc_knl_build;
  bnd->knl_run = ispc_knl_run;
  bnd->knl_free = ispc_knl_free;
  bnd->finalize = ispc_finalize;

  return 0;
}
