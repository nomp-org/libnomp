#include "nomp-impl.h"
#include "nomp-jit.h"
#include <ispcrt/ispcrt.h>

static const char *ERR_STR_ISPC_FAILURE =
    "ISPC %s failed with error message: %s.";

/**
 * @ingroup nomp_types
 *
 * @brief Represents a memory block that can be used for data storage and
 * transfer between a host and a device.
 */
struct ispc_backend {
  /**
   * @brief Path to the ISPC compiler.
   */
  char *ispc_cc;
  /**
   * @brief Path to the C/C++ compiler.
   */
  char *cc;
  /**
   * @brief ISPC compiler flags.
   */
  char *ispc_flags;
  /**
   * @brief C/C++ compiler flags.
   */
  char *cc_flags;
  /**
   * @brief Handle to the selected device.
   */
  ISPCRTDevice device;
  /**
   * @brief Type of device.
   */
  ISPCRTDeviceType device_type;
  /**
   * @brief Handle to the task queue.
   */
  ISPCRTTaskQueue queue;
  /**
   * @brief Flags to configure the creation of new memory views.
   */
  ISPCRTNewMemoryViewFlags flags;
};

/**
 * @ingroup nomp_types
 *
 * @brief Struct to store ISPC program information.
 */
struct ispc_prog {
  /**
   * @brief ID of the ISPC program.
   */
  int ispc_id;
  /**
   * @brief Module of the ISPC program.
   */
  ISPCRTModule module;
  /**
   * @brief Kernel of the ISPC program.
   */
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
      return nomp_log(NOMP_ISPC_FAILURE, NOMP_ERROR, ERR_STR_ISPC_FAILURE,     \
                      msg, err_message);                                       \
    }                                                                          \
  }

static int ispc_update(struct nomp_backend_t *bnd, struct nomp_mem_t *m,
                       const nomp_map_direction_t op, size_t start, size_t end,
                       size_t usize) {
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;

  if (op & NOMP_ALLOC) {
    ISPCRTMemoryView view =
        ispcrtNewMemoryView(ispc->device, m->hptr,
                            NOMP_MEM_BYTES(start, end, usize), &(ispc->flags));
    chk_ispcrt("memory allocation", rt_error);
    m->bptr = view;
  }

  if (op & NOMP_TO) {
    ispcrtCopyToDevice(ispc->queue, (ISPCRTMemoryView)(m->bptr));
    chk_ispcrt("memory copy to device", rt_error);
  } else if (op == NOMP_FROM) {
    ispcrtCopyToHost(ispc->queue, (ISPCRTMemoryView)(m->bptr));
    chk_ispcrt("memory copy from device", rt_error);
  } else if (op == NOMP_FREE) {
    ispcrtRelease((ISPCRTMemoryView)(m->bptr));
    m->bptr = NULL;
    chk_ispcrt("memory freeing", rt_error);
  }

  return 0;
}

static int ispc_knl_build(struct nomp_backend_t *bnd, struct nomp_prog_t *prg,
                          const char *source, const char *name) {
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;
  struct ispc_prog *iprg = nomp_calloc(struct ispc_prog, 1);

  char cwd[BUFSIZ];
  if (getcwd(cwd, BUFSIZ) == NULL) {
    return nomp_log(NOMP_JIT_FAILURE, NOMP_ERROR, ERR_STR_ISPC_FAILURE,
                    "get cwd");
  }

  const char *src_f = "nomp_ispc.ispc", *dev_f = "nomp_ispc.dev.o";
  const char *lib = "nomp_ispc";
  char *wkdir = nomp_str_cat(3, BUFSIZ, cwd, "/", ".nomp_jit_cache");
  int err = nomp_jit_compile(NULL, source, ispc->ispc_cc, ispc->ispc_flags,
                             NULL, wkdir, src_f, dev_f);
  if (err) {
    nomp_free(&wkdir);
    return err;
  }

  char *lib_so = nomp_str_cat(3, BUFSIZ, "lib", lib, ".so");
  err = nomp_jit_compile(&iprg->ispc_id, source, ispc->cc, ispc->cc_flags, name,
                         wkdir, dev_f, lib_so);
  nomp_free(&wkdir), nomp_free(&lib_so);
  nomp_check(err);
  prg->bptr = (void *)iprg;
  return 0;
}

static int ispc_knl_run(struct nomp_backend_t *bnd, struct nomp_prog_t *prg) {
  const int ndim = prg->ndim, nargs = prg->nargs;
  struct nomp_arg_t *args = prg->args;
  size_t *global = prg->global;

  void *vargs[NOMP_MAX_KNL_ARGS];
  for (int i = 0; i < nargs; i++) {
    if (args[i].type == NOMP_PTR)
      vargs[i] = ispcrtDevicePtr((ISPCRTMemoryView)(args[i].ptr));
    else
      vargs[i] = args[i].ptr;
  }

  int one = 1;
  for (int d = 0; d < ndim; d++)
    vargs[nargs + d] = (void *)&(global[d]);
  for (int d = ndim; d < 3; d++)
    vargs[nargs + d] = (void *)&one;

  struct ispc_prog *iprg = (struct ispc_prog *)prg->bptr;
  return nomp_jit_run(iprg->ispc_id, vargs);
}

static int ispc_knl_free(struct nomp_prog_t *prg) {
  struct ispc_prog *iprg = (struct ispc_prog *)prg->bptr;

  if (iprg)
    return nomp_jit_free(&iprg->ispc_id);
  return 0;
}

static int ispc_finalize(struct nomp_backend_t *bnd) {
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;

  if (ispc) {
    ispcrtRelease(ispc->device);
    chk_ispcrt("device release", rt_error);
    ispcrtRelease(ispc->queue);
    chk_ispcrt("queue release", rt_error);
    nomp_free(&ispc->cc), nomp_free(&ispc->cc_flags);
    nomp_free(&ispc->ispc_cc), nomp_free(&ispc->ispc_flags);
  }
  nomp_free(&bnd->bptr);
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
    return nomp_log(NOMP_ISPC_FAILURE, NOMP_ERROR,
                    "Environment variable NOMP_CC is not set. Please set "
                    "it to point to host compiler.");
  }

  tmp = getenv("NOMP_ISPC_CC");
  if (tmp) {
    size_t size;
    nomp_check(nomp_path_len(&size, tmp));
    ispc->ispc_cc = strndup(tmp, size + 1);
  } else {
    return nomp_log(NOMP_ISPC_FAILURE, NOMP_ERROR,
                    "Environment variable NOMP_ISPC_CC is not set. Please "
                    "set it to point to ISPC compiler.");
  }

  tmp = getenv("NOMP_CFLAGS");
  if (tmp) {
    ispc->cc_flags = strndup(tmp, NOMP_MAX_BUFSIZ + 1);
  } else {
    return nomp_log(NOMP_ISPC_FAILURE, NOMP_ERROR,
                    "Environment variable NOMP_CFLAGS is not set. Please "
                    "set it with suitable to host compiler flags.");
  }

  tmp = getenv("NOMP_ISPC_CFLAGS");
  if (tmp) {
    ispc->ispc_flags = strndup(tmp, NOMP_MAX_BUFSIZ + 1);
  } else {
    return nomp_log(NOMP_ISPC_FAILURE, NOMP_ERROR,
                    "Environment variable NOMP_ISPC_CFLAGS is not set. "
                    "Please set it with suitable ISPC compiler flags.");
  }

  return 0;
}

static int ispc_sync(struct nomp_backend_t *bnd) { return 0; }

int ispc_init(struct nomp_backend_t *bnd, const int platform_type,
              const int device_id) {
  ispcrtSetErrorFunc(ispcrt_error);
  if (platform_type < 0 | platform_type >= 2) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Platform type %d provided to libnomp is not valid.",
                    platform_type);
  }
  uint32_t num_devices =
      ispcrtGetDeviceCount(nomp_to_ispc_device[platform_type]);
  chk_ispcrt("get device count", rt_error);
  if (device_id < 0 || device_id >= num_devices) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
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
