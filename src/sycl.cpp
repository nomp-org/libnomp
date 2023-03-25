#include "nomp-impl.h"
#include "nomp-jit.h"
#include <CL/sycl.hpp>

static const char *ERR_STR_SYCL_FAILURE = "SYCL backend failed with error: %s.";

#define chk_cl(call, msg)                                                      \
  {                                                                            \
    cl_int err = (call);                                                       \
    if (err != CL_SUCCESS) {                                                   \
      return nomp_set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR,                     \
                          ERR_STR_OPENCL_FAILURE, msg, err);                   \
    }                                                                          \
  }

struct sycl_backend {
  sycl::device device_id;
  sycl::queue queue;
  sycl::context ctx;
  const char *compiler;
  const char *compiler_flags;
};

struct sycl_prog {
  int sycl_id;
};

static int sycl_update(struct backend *bnd, struct mem *m, const int op) {
  struct sycl_backend *sycl = (struct sycl_backend *)bnd->bptr;

  if (op & NOMP_ALLOC) {
    try {
      m->bptr = sycl::malloc_device((m->idx1 - m->idx0) * m->usize,
                                    sycl->device_id, sycl->ctx);
    } catch (const std::exception &ex) {
      return nomp_set_log(NOMP_SYCL_FAILURE, NOMP_ERROR, ERR_STR_SYCL_FAILURE,
                          ex.what());
    }
  }

  if (op & NOMP_TO) {
    try {
      sycl->queue.memcpy(m->bptr,
                         static_cast<char *>(m->hptr) + m->usize * m->idx0,
                         (m->idx1 - m->idx0) * m->usize);
      sycl->queue.wait();
    } catch (const std::exception &ex) {
      return nomp_set_log(NOMP_SYCL_FAILURE, NOMP_ERROR, ERR_STR_SYCL_FAILURE,
                          ex.what());
    }
  } else if (op == NOMP_FROM) {
    try {
      sycl->queue.memcpy(static_cast<char *>(m->hptr) + m->usize * m->idx0,
                         m->bptr, (m->idx1 - m->idx0) * m->usize);
      sycl->queue.wait();
    } catch (const std::exception &ex) {
      return nomp_set_log(NOMP_SYCL_FAILURE, NOMP_ERROR, ERR_STR_SYCL_FAILURE,
                          ex.what());
    }
  } else if (op == NOMP_FREE) {
    try {
      sycl::free(m->bptr, sycl->ctx);
      m->bptr = NULL;
    } catch (const std::exception &ex) {
      return nomp_set_log(NOMP_SYCL_FAILURE, NOMP_ERROR, ERR_STR_SYCL_FAILURE,
                          ex.what());
    }
  }

  return 0;
}

static int sycl_knl_free(struct prog *prg) {
  struct sycl_prog *sycl_prg = (struct sycl_prog *)prg->bptr;
  int err = jit_free(&sycl_prg->sycl_id);
  nomp_free(prg->bptr), prg->bptr = NULL;

  return err;
}

static int sycl_knl_build(struct backend *bnd, struct prog *prg,
                          const char *source, const char *name) {
  struct sycl_backend *sycl = (struct sycl_backend *)bnd->bptr;
  prg->bptr = nomp_calloc(struct sycl_prog, 1);
  struct sycl_prog *sycl_prg = (struct sycl_prog *)prg->bptr;
  sycl_prg->sycl_id = -1;

  char cwd[BUFSIZ];
  if (getcwd(cwd, BUFSIZ) == NULL) {
    return nomp_set_log(NOMP_SYCL_FAILURE, NOMP_ERROR, ERR_STR_SYCL_FAILURE,
                        "Failed to get current working directory.");
  }

  char *wkdir = nomp_str_cat(3, BUFSIZ, cwd, "/", ".nomp_jit_cache");
  int err = jit_compile(&sycl_prg->sycl_id, source, sycl->compiler,
                        sycl->compiler_flags, name, wkdir);
  nomp_free(wkdir);

  return err;
}

static int sycl_knl_run(struct backend *bnd, struct prog *prg, va_list args) {
  struct sycl_backend *sycl = (struct sycl_backend *)bnd->bptr;
  struct sycl_prog *sycl_prg = (struct sycl_prog *)prg->bptr;
  struct mem *m;
  size_t size;
  void *arg_list[prg->nargs + 2];
  int err = 1;
  for (int i = 0; i < prg->nargs; i++) {
    const char *var = va_arg(args, const char *);
    int type = va_arg(args, int);
    size = va_arg(args, size_t);
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
      p = m->bptr;
      break;
    default:;
      return nomp_set_log(
          NOMP_USER_KNL_ARG_TYPE_IS_INVALID, NOMP_ERROR,
          "Kernel argument type %d passed to libnomp is not valid.", type);
      break;
    }
    arg_list[i] = p;
  }

  arg_list[prg->nargs] = (void *)&sycl->queue;

  size_t global[3];
  for (unsigned i = 0; i < prg->ndim; i++)
    global[i] = prg->global[i] * prg->local[i];

  if (prg->ndim == 1) {
    sycl::range global_range = sycl::range(global[0]);
    sycl::range local_range = sycl::range(prg->local[0]);
    sycl::nd_range<1> nd_range = sycl::nd_range(global_range, local_range);
    arg_list[prg->nargs + 1] = (void *)&nd_range;
    err = jit_run(sycl_prg->sycl_id, arg_list);
  } else if (prg->ndim == 2) {
    sycl::range global_range = sycl::range(global[0], global[1]);
    sycl::range local_range = sycl::range(prg->local[0], prg->local[1]);
    sycl::nd_range<2> nd_range = sycl::nd_range(global_range, local_range);
    arg_list[prg->nargs + 1] = (void *)&nd_range;
    err = jit_run(sycl_prg->sycl_id, arg_list);
  } else if (prg->ndim == 3) {
    sycl::range global_range = sycl::range(global[0], global[1], global[2]);
    sycl::range local_range =
        sycl::range(prg->local[0], prg->local[1], prg->local[2]);
    sycl::nd_range<3> nd_range = sycl::nd_range(global_range, local_range);
    arg_list[prg->nargs + 1] = (void *)&nd_range;
    err = jit_run(sycl_prg->sycl_id, arg_list);
  }

  return err;
}

static int sycl_sync(struct backend *bnd) {
  struct sycl_backend *sycl = (struct sycl_backend *)bnd->bptr;
  sycl->queue.wait();
  return 0;
}

static int sycl_finalize(struct backend *bnd) {
  int err = 0;
  struct sycl_backend *sycl = (struct sycl_backend *)bnd->bptr;
  nomp_free(bnd->bptr), bnd->bptr = NULL;

  return err;
}

int sycl_init(struct backend *bnd, const int platform_id, const int device_id) {
  bnd->bptr = nomp_calloc(struct sycl_backend, 1);
  struct sycl_backend *sycl = (struct sycl_backend *)bnd->bptr;

  char *tmp = getenv("ICPX_COMPILER_PATH");
  if (tmp)
    sycl->compiler = strndup(tmp, MAX_BUFSIZ), nomp_free(tmp);
  else
    sycl->compiler = "/opt/intel/oneapi/compiler/2023.0.0/linux/bin/icpx";
  if ((tmp = getenv("ICPX_COMPILER_FLAGS")))
    sycl->compiler_flags = strndup(tmp, MAX_BUFSIZ), nomp_free(tmp);
  else
    sycl->compiler_flags = "-fsycl -fPIC -shared";

  sycl::platform sycl_platform = sycl::platform();
  auto sycl_pplatforms = sycl_platform.get_platforms();

  if (platform_id < 0 | platform_id >= sycl_pplatforms.size()) {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "Platform id %d provided to libnomp is not valid.",
                        platform_id);
  }
  sycl_platform = sycl_pplatforms[platform_id];
  auto sycl_pdevices = sycl_platform.get_devices();

  if (device_id < 0 || device_id >= sycl_pdevices.size()) {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        ERR_STR_USER_DEVICE_IS_INVALID, device_id);
  }

  sycl->device_id = sycl_pdevices[device_id];
  sycl->ctx = sycl::context(sycl->device_id);
  sycl->queue = sycl::queue(sycl->ctx, sycl->device_id);

  bnd->update = sycl_update;
  bnd->knl_build = sycl_knl_build;
  bnd->knl_run = sycl_knl_run;
  bnd->knl_free = sycl_knl_free;
  bnd->sync = sycl_sync;
  bnd->finalize = sycl_finalize;

  return 0;
}
