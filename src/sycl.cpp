#include "nomp-impl.h"
#include "nomp-jit.h"
#include <CL/opencl.h>
#include <CL/sycl.hpp>
#include <dlfcn.h>

// TODO: Handle errors properly in SYCL backend
struct sycl_backend {
  sycl::device device_id;
  sycl::queue queue;
  sycl::context ctx;
  int sycl_id = -1;
};

static int sycl_update(struct backend *bnd, struct mem *m, const int op) {
  struct sycl_backend *sycl = (struct sycl_backend *)bnd->bptr;

  if (op & NOMP_ALLOC) {
    // TODO: check and handle errors
    m->bptr = sycl::malloc_device((m->idx1 - m->idx0) * m->usize,
                                  sycl->device_id, sycl->ctx);
  }

  if (op & NOMP_TO) {
    // TODO: check and handle errors
    sycl->queue.memcpy(m->bptr,
                       static_cast<char *>(m->hptr) + m->usize * m->idx0,
                       (m->idx1 - m->idx0) * m->usize);
    sycl->queue.wait();
  }

  if (op == NOMP_FROM) {
    // TODO: check and handle errors
    sycl->queue.memcpy(static_cast<char *>(m->hptr) + m->usize * m->idx0,
                       m->bptr, (m->idx1 - m->idx0) * m->usize);
    sycl->queue.wait();
  } else if (op == NOMP_FREE) {
    // TODO: check and handle errors
    // TODO: use nomp_free instead if possible
    sycl::free(m->bptr, sycl->ctx);
    m->bptr = NULL;
  }

  return 0;
}

static int sycl_knl_free(struct prog *prg) {
  nomp_free(prg->bptr), prg->bptr = NULL;
  return 0;
}

static int sycl_knl_build(struct backend *bnd, struct prog *prg,
                          const char *source, const char *name) {
  struct sycl_backend *sycl = (sycl_backend *)bnd->bptr;
  int err;

  char cwd[BUFSIZ];
  // TODO: use nomp_check() instead
  if (getcwd(cwd, BUFSIZ) == NULL) {
    printf("Error in cwd");
    return 0;
  }

  char *wkdir = nomp_str_cat(3, BUFSIZ, cwd, "/", ".nomp_jit_cache");
  // TODO: make icpx path generic
  err = jit_compile(&sycl->sycl_id, source,
                    "/opt/intel/oneapi/compiler/2023.0.0/linux/bin/icpx",
                    "-fsycl -fPIC -shared", "kernel_function", wkdir);
  // TODO: use nomp_set_log() to handle err
  return 0;
}

static int sycl_knl_run(struct backend *bnd, struct prog *prg, va_list args) {
  struct sycl_backend *sycl = (sycl_backend *)bnd->bptr;
  struct mem *m;
  size_t size;
  sycl->queue = sycl::queue(sycl->ctx, sycl->device_id);
  void *arg_list[prg->nargs + 2];
  int err;
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
      if (m == NULL)
        return nomp_set_log(NOMP_USER_MAP_PTR_IS_INVALID, NOMP_ERROR,
                            ERR_STR_USER_MAP_PTR_IS_INVALID, p);
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
    err = jit_run(sycl->sycl_id, arg_list);
  } else if (prg->ndim == 2) {
    sycl::range global_range = sycl::range(global[0], global[1]);
    sycl::range local_range = sycl::range(prg->local[0], prg->local[1]);
    sycl::nd_range<2> nd_range = sycl::nd_range(global_range, local_range);
    arg_list[prg->nargs + 1] = (void *)&nd_range;
    err = jit_run(sycl->sycl_id, arg_list);
  } else if (prg->ndim == 3) {
    sycl::range global_range = sycl::range(global[0], global[1], global[2]);
    sycl::range local_range =
        sycl::range(prg->local[0], prg->local[1], prg->local[2]);
    sycl::nd_range<3> nd_range = sycl::nd_range(global_range, local_range);
    arg_list[prg->nargs + 1] = (void *)&nd_range;
    err = jit_run(sycl->sycl_id, arg_list);
  }
  // TODO: use nomp_set_log() to handle err
  return err;
}

static int sycl_finalize(struct backend *bnd) {
  int err;
  struct sycl_backend *sycl = (sycl_backend *)bnd->bptr;
  err = jit_free(&sycl->sycl_id);
  // TODO: use nomp_set_log() to handle err
  nomp_free(bnd->bptr), bnd->bptr = NULL;
  return 0;
}

int sycl_init(struct backend *bnd, const int platform_id, const int device_id) {
  bnd->bptr = nomp_calloc(struct sycl_backend, 1);
  struct sycl_backend *sycl = (sycl_backend *)bnd->bptr;

  sycl::platform sycl_platform = sycl::platform();
  auto sycl_pplatforms = sycl_platform.get_platforms();

  if (platform_id < 0 | platform_id >= sycl_pplatforms.size())
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "Platform id %d provided to libnomp is not valid.",
                        platform_id);
  sycl_platform = sycl_pplatforms[platform_id];
  auto sycl_pdevices = sycl_platform.get_devices();

  if (device_id < 0 || device_id >= sycl_pdevices.size())
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        ERR_STR_USER_DEVICE_IS_INVALID, device_id);

  sycl::device sycl_device = sycl_pdevices[device_id];
  sycl->device_id = sycl_device;
  sycl::context sycl_ctx = sycl::context(sycl_device);

  sycl->ctx = sycl_ctx;
  sycl::queue sycl_queue = sycl::queue(sycl_ctx, sycl_device);
  sycl->queue = sycl_queue;

  bnd->update = sycl_update;
  bnd->knl_build = sycl_knl_build;
  bnd->knl_run = sycl_knl_run;
  bnd->knl_free = sycl_knl_free;
  bnd->finalize = sycl_finalize;
  return 0;
}
