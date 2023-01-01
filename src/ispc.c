#include "ispcrt.h"
#include "nomp-impl.h"

// TODO: Handle errors properly in ISPC backend

#define NARGS_MAX 64

struct mem_view {
  struct mem *m;
  void *view;
};

struct ispc_backend {
  ISPCRTDevice device;
  ISPCRTTaskQueue queue;
};

struct ispc_prog {
  ISPCRTModule module;
  ISPCRTKernel kernel;
};

static struct mem_view **mem_views = NULL;
static int mems_n = 0;
static int mems_max = 0;

static unsigned mem_if_exist(void *p, size_t idx0, size_t idx1) {
  for (unsigned i = 0; i < mems_n; i++) {
    if (mem_views[i] && mem_views[i]->m && mem_views[i]->m->hptr == p &&
        mem_views[i]->m->idx0 == idx0 && mem_views[i]->m->idx1 == idx1)
      return i;
  }
  return mems_n;
}

static ISPCRTError rt_error = ISPCRT_NO_ERROR;
static char *err_message = NULL;
static void ispcrt_error(ISPCRTError err_code, const char *message) {
  rt_error = err_code;
  err_message = (char *)message;
}

static int ispc_update(struct backend *bnd, struct mem *m, const int op) {
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;

  if (op & NOMP_ALLOC) {
    ISPCRTNewMemoryViewFlags flags;
    flags.allocType = ISPCRT_ALLOC_TYPE_DEVICE;
    ISPCRTMemoryView view = ispcrtNewMemoryView(
        ispc->device, &m->hptr, (m->idx1 - m->idx0) * m->usize, &flags);
    m->bptr = ispcrtDevicePtr(view);

    unsigned idx = mem_if_exist(&m->hptr, m->idx0, m->idx1);
    if (idx == mems_n) {
      if (mems_n == mems_max) {
        mems_max += mems_max / 2 + 1;
        mem_views = trealloc(mem_views, struct mem_view *, mems_max);
      }
      struct mem_view *m_view = mem_views[mems_n] = tcalloc(struct mem_view, 1);
      m_view->m = m;
    }
    mem_views[idx]->view = m->bptr;
  }

  if (op & NOMP_TO) {
    // Objects which used as inputs for ISPC kernel should be
    // explicitly copied to device from host
    unsigned idx = mem_if_exist(&m->hptr, m->idx0, m->idx1);
    if (idx != mems_n)
      ispcrtCopyToDevice(ispc->queue, mem_views[idx]->view);
  }

  if (op == NOMP_FROM) {
    unsigned idx = mem_if_exist(&m->hptr, m->idx0, m->idx1);
    if (idx != mems_n)
      ispcrtCopyToHost(ispc->queue, mem_views[idx]->view);
  } else if (op == NOMP_FREE) {
    m->bptr = NULL;
  }

  return 0;
}

static int ispc_knl_build(struct backend *bnd, struct prog *prg,
                          const char *source, const char *name) {
  struct ispc_backend *ispc = bnd->bptr;
  struct ispc_prog *ispc_prg = prg->bptr = tcalloc(struct ispc_prog, 1);

  // TODO: copy the content to a file and build

  // Create module and kernel to execute
  ISPCRTModuleOptions options = {};
  ispc_prg->module = ispcrtLoadModule(ispc->device, name, options);
  if (rt_error != ISPCRT_NO_ERROR) {
    ispc_prg->module = NULL;
    return rt_error;
  }

  ispc_prg->kernel = ispcrtNewKernel(ispc->device, ispc_prg->module, name);
  if (rt_error != ISPCRT_NO_ERROR) {
    ispc_prg->module = NULL;
    ispc_prg->kernel = NULL;
    return rt_error;
  }
  return 0;
}

static int ispc_knl_run(struct backend *bnd, struct prog *prg, va_list args) {
  const int ndim = prg->ndim, nargs = prg->nargs;
  const size_t *global = prg->global;
  size_t num_bytes = 0;

  struct mem *m;
  void *vargs[NARGS_MAX];
  for (int i = 0; i < nargs; i++) {
    const char *var = va_arg(args, const char *);
    int type = va_arg(args, int);
    size_t size = va_arg(args, size_t);
    num_bytes += size;
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

  ISPCRTNewMemoryViewFlags flags;
  flags.allocType = ISPCRT_ALLOC_TYPE_DEVICE;
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;
  ISPCRTMemoryView params =
      ispcrtNewMemoryView(ispc->device, vargs, num_bytes, &flags);
  ispcrtCopyToDevice(ispc->queue, params);

  // launch kernel
  struct ispc_prog *ispc_prg = (struct ispc_prog *)prg->bptr;
  // TODO:
  ispcrtLaunch1D(ispc->queue, ispc_prg->kernel, params, global[0]);
  ispcrtSync(ispc->queue);
  return rt_error != ISPCRT_NO_ERROR;
}

static int ispc_knl_free(struct prog *prg) {
  struct ispc_prog *iprg = (struct ispc_prog *)prg->bptr;
  ispcrtRelease(iprg->kernel);
  ispcrtRelease(iprg->module);
  return 0;
}

static int ispc_finalize(struct backend *bnd) {
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;
  ispcrtRelease(ispc->device);
  ispcrtRelease(ispc->queue);
  tfree(bnd->bptr), bnd->bptr = NULL;
  return 0;
}

int ispc_init(struct backend *bnd, const int platform_type,
              const int device_id) {
  ispcrtSetErrorFunc(ispcrt_error);
  if (platform_type < 0 | platform_type >= 3)
    return set_log(NOMP_USER_PLATFORM_IS_INVALID, NOMP_ERROR,
                   "Platform type %d provided to libnomp is not valid.",
                   platform_type);
  uint32_t num_devices = ispcrtGetDeviceCount(platform_type);
  if (rt_error != ISPCRT_NO_ERROR)
    return rt_error;
  if (device_id < 0 || device_id >= num_devices)
    return rt_error;
  ISPCRTDevice device = ispcrtGetDevice(platform_type, device_id);
  if (rt_error != ISPCRT_NO_ERROR)
    return rt_error;

  bnd->bptr = tcalloc(struct ispc_backend, 1);
  struct ispc_backend *ispc = (struct ispc_backend *)bnd->bptr;
  ispc->device = device;
  ispc->queue = ispcrtNewTaskQueue(device);
  if (rt_error != ISPCRT_NO_ERROR)
    return rt_error;

  bnd->update = ispc_update;
  bnd->knl_build = ispc_knl_build;
  bnd->knl_run = ispc_knl_run;
  bnd->knl_free = ispc_knl_free;
  bnd->finalize = ispc_finalize;

  return 0;
}
