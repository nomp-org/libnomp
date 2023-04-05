#include "nomp-impl.h"
#include "nomp-jit.h"
#include <ispcrt/ispcrt.h>

#define NARGS_MAX 64

static const char *ERR_STR_ISPC_FAILURE = "ISPC %s failed with error code: %d.";

struct ispc_backend {
  char *ispc_cc, *cc, *ispc_flags, *cc_flags;
  ISPCRTDevice device;
  ISPCRTDeviceType device_type;
  ISPCRTTaskQueue queue;
  ISPCRTNewMemoryViewFlags flags;
};

struct ispc_prog {
  int ispc_id;
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
    if (x != ISPCRT_NO_ERROR) {                                                \
      return nomp_set_log(NOMP_ISPC_FAILURE, NOMP_ERROR, ERR_STR_ISPC_FAILURE, \
                          msg, err_message);                                   \
    }                                                                          \
  }

static int ispc_update(struct backend *bnd, struct mem *m, const int op) {
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;

  if (op & NOMP_ALLOC) {
    ISPCRTMemoryView view = ispcrtNewMemoryView(
        ispc->device, m->hptr, (m->idx1 - m->idx0) * m->usize, &(ispc->flags));
    chk_ispcrt("error in alloc", rt_error);
    m->bptr = view;
  }

  if (op & NOMP_TO) {
    ispcrtCopyToDevice(ispc->queue, (ISPCRTMemoryView)(m->bptr));
    chk_ispcrt("error in copy to device", rt_error);
  } else if (op == NOMP_FROM) {
    ispcrtCopyToHost(ispc->queue, (ISPCRTMemoryView)(m->bptr));
    chk_ispcrt("error in copy to host", rt_error);
  } else if (op == NOMP_FREE) {
    ispcrtRelease((ISPCRTMemoryView)(m->bptr));
    m->bptr = NULL;
    chk_ispcrt("error in free", rt_error);
  }

  return 0;
}

static int ispc_knl_build(struct backend *bnd, struct prog *prg,
                          const char *source, const char *name) {
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;
  struct ispc_prog *iprg = nomp_calloc(struct ispc_prog, 1);

  char cwd[BUFSIZ];
  if (getcwd(cwd, BUFSIZ) == NULL) {
    return nomp_set_log(NOMP_JIT_FAILURE, NOMP_ERROR, ERR_STR_ISPC_FAILURE,
                        "get cwd");
  }

  const char *src_f = "nomp_ispc.ispc", *dev_f = "nomp_ispc.dev.o";
  const char *lib = "nomp_ispc";
  char *wkdir = nomp_str_cat(3, BUFSIZ, cwd, "/", ".nomp_jit_cache");
  int err = jit_compile(NULL, source, ispc->ispc_cc, ispc->ispc_flags, NULL,
                        wkdir, src_f, dev_f);
  if (err) {
    nomp_free(wkdir);
    return err;
  }

  char *lib_so = nomp_str_cat(3, BUFSIZ, "lib", lib, ".so");
  err = jit_compile(&iprg->ispc_id, source, ispc->cc, ispc->cc_flags, name,
                    wkdir, dev_f, lib_so);
  nomp_free(wkdir), nomp_free(lib_so);
  nomp_check(err);
  prg->bptr = (void *)iprg;
  return 0;
}

static int ispc_knl_run(struct backend *bnd, struct prog *prg, va_list args) {
  const int ndim = prg->ndim, nargs = prg->nargs;
  size_t *global = prg->global;

  struct mem *m;
  void *vargs[NARGS_MAX];
  for (int i = 0; i < nargs; i++) {
    const char *var = va_arg(args, const char *);
    int type = va_arg(args, int);
    size_t size = va_arg(args, size_t);
    void *p = va_arg(args, void *);
    switch (type) {
    case NOMP_INT:
    case NOMP_UINT:
    case NOMP_FLOAT:
      break;
    case NOMP_PTR:
      m = mem_if_mapped(p);
      if (m == NULL) {
        return nomp_set_log(NOMP_USER_MAP_PTR_IS_INVALID, NOMP_ERROR,
                            ERR_STR_USER_MAP_PTR_IS_INVALID, p);
      }
      p = ispcrtDevicePtr((ISPCRTMemoryView)(m->bptr));
      break;
    default:
      return nomp_set_log(NOMP_USER_KNL_ARG_TYPE_IS_INVALID, NOMP_ERROR,
                          "Kernel argument type %d is not valid.", type);
      break;
    }
    vargs[i] = p;
  }

  int one = 1;
  for (int d = 0; d < ndim; d++)
    vargs[nargs + d] = (void *)&(global[d]);
  for (int d = ndim; d < 3; d++)
    vargs[nargs + d] = (void *)&one;

  struct ispc_prog *iprg = (struct ispc_prog *)prg->bptr;
  return jit_run(iprg->ispc_id, vargs);
}

static int ispc_knl_free(struct prog *prg) {
  struct ispc_prog *iprg = (struct ispc_prog *)prg->bptr;
  return jit_free(&iprg->ispc_id);
}

static int ispc_finalize(struct backend *bnd) {
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;
  ispcrtRelease(ispc->device);
  chk_ispcrt("device release", rt_error);
  ispcrtRelease(ispc->queue);
  chk_ispcrt("queue release", rt_error);
  nomp_free(ispc->cc), nomp_free(ispc->cc_flags);
  nomp_free(ispc->ispc_cc), nomp_free(ispc->ispc_flags);
  nomp_free(bnd->bptr), bnd->bptr = NULL;
  return 0;
}

static int nomp_to_ispc_device[2] = {ISPCRT_DEVICE_TYPE_CPU,
                                     ISPCRT_DEVICE_TYPE_GPU};

static int ispc_chk_env(struct ispc_backend *ispc) {
  char *tmp = getenv("NOMP_CC");
  if (tmp) {
    size_t size;
    nomp_check(nomp_path_len(&size, tmp));
    ispc->cc = strndup(tmp, size + 1);
  } else {
    return nomp_set_log(NOMP_ISPC_FAILURE, NOMP_ERROR,
                        "CC compiler NOMP_CC must be set.");
  }

  tmp = getenv("NOMP_ISPC_CC");
  if (tmp) {
    size_t size;
    nomp_check(nomp_path_len(&size, tmp));
    ispc->ispc_cc = strndup(tmp, size + 1);
  } else {
    return nomp_set_log(NOMP_ISPC_FAILURE, NOMP_ERROR,
                        "ISPC compiler NOMP_ISPC_CC must be set.");
  }

  tmp = getenv("NOMP_CFLAGS");
  if (tmp) {
    ispc->cc_flags = strndup(tmp, MAX_BUFSIZ + 1);
  } else {
    return nomp_set_log(NOMP_ISPC_FAILURE, NOMP_ERROR,
                        "CC compiler flags NOMP_CFLAGS must be set.");
  }

  tmp = getenv("NOMP_ISPC_CFLAGS");
  if (tmp) {
    ispc->ispc_flags = strndup(tmp, MAX_BUFSIZ + 1);
  } else {
    return nomp_set_log(NOMP_ISPC_FAILURE, NOMP_ERROR,
                        "ISPC compiler flags NOMP_ISPC_CFLAGS must be set.");
  }

  return 0;
}

static int ispc_sync(struct backend *bnd) { return 0; }

int ispc_init(struct backend *bnd, const int platform_type,
              const int device_id) {
  ispcrtSetErrorFunc(ispcrt_error);
  if (platform_type < 0 | platform_type >= 2) {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "Platform type %d provided to libnomp is not valid.",
                        platform_type);
  }
  uint32_t num_devices =
      ispcrtGetDeviceCount(nomp_to_ispc_device[platform_type]);
  chk_ispcrt("get device count", rt_error);
  if (device_id < 0 || device_id >= num_devices) {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }
  ISPCRTDevice device = ispcrtGetDevice(platform_type, device_id);
  chk_ispcrt("device get", rt_error);

  struct ispc_backend *ispc = bnd->bptr = nomp_calloc(struct ispc_backend, 1);
  ispc->flags.allocType = ISPCRT_ALLOC_TYPE_DEVICE;
  ispc->device = device;
  ispc->device_type = platform_type;
  ispc->queue = ispcrtNewTaskQueue(device);
  chk_ispcrt("context create", rt_error);
  nomp_check(ispc_chk_env(ispc));

  bnd->update = ispc_update;
  bnd->knl_build = ispc_knl_build;
  bnd->knl_run = ispc_knl_run;
  bnd->knl_free = ispc_knl_free;
  bnd->finalize = ispc_finalize;
  bnd->sync = ispc_sync;

  return 0;
}
