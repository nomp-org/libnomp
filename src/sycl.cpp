
#include "nomp-impl.h"
#include <CL/opencl.h>
#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>
#include <dlfcn.h>
// TODO: Handle errors properly in SYCL backend
struct sycl_backend {
  sycl::device device_id;
  sycl::queue queue;
  sycl::context ctx;
};
// auto mybundle = sycl::make_kernel_bundle<sycl::backend::opencl,
// sycl::bundle_state::executable>(ocl_program, ctx);


static int sycl_update(struct backend *bnd, struct mem *m, const int op) {
  struct sycl_backend *sycl = (struct sycl_backend *)bnd->bptr;

  if (op & NOMP_ALLOC) {
    m->bptr = sycl::malloc_device((m->idx1 - m->idx0) * m->usize, sycl->device_id,sycl->ctx);
    //sycl->queue.wait();
  }

  if (op & NOMP_TO) {
    sycl->queue.memcpy(m->bptr, m->hptr, (m->idx1 - m->idx0) * m->usize);
    sycl->queue.wait();
    // for (unsigned i = m->idx0; i < m->idx1; i++)
    //   printf("NOMP_TO %i , %i\n",((int *)m->bptr)[i],((int *)m->hptr)[i]);
  }

  if (op == NOMP_FROM) {
    sycl->queue.memcpy(m->hptr, m->bptr, (m->idx1 - m->idx0) * m->usize);
    sycl->queue.wait();

    // for (unsigned i = m->idx0; i < m->idx1; i++)
    //   printf("NOMP_FROM %i , %i\n",((int *)m->hptr)[i],((int *)m->bptr)[i]);

  } else if (op == NOMP_FREE) {
    sycl::free(m->bptr, sycl->ctx);
    m->bptr = NULL;
  }

  return 0;
}

static int sycl_knl_free(struct prog *prg) {
  //struct opencl_prog *ocl_prg = (opencl_prog *)prg->bptr;
  //tfree(prg->bptr), prg->bptr = NULL;

  return 0;
}

static int sycl_knl_build(struct backend *bnd, struct prog *prg,
                          const char *source, const char *name) {
  struct sycl_backend *sycl = (sycl_backend *)bnd->bptr;

  char *path=writefile(bnd->knl_fun);
  compile(path);
 // prg->bptr = tcalloc(struct opencl_prog, 1);
 // struct opencl_prog *ocl_prg = (opencl_prog *)prg->bptr;
  
  //printf("source  %s\n",source);
  return 0;
}

static int sycl_knl_run(struct backend *bnd, struct prog *prg, va_list args) {
  struct sycl_backend *sycl = (sycl_backend *)bnd->bptr;
  //struct opencl_prog *ocl_prg = (struct opencl_prog *)prg->bptr;
  struct mem *m;
  size_t size;
  sycl->queue = sycl::queue(sycl->ctx, sycl->device_id);
  void *arg_list[prg->nargs];
  int err;
  for (int i = 0; i < prg->nargs; i++) {
    const char *var = va_arg(args, const char *);
    int type = va_arg(args, int);
    size = va_arg(args, size_t);
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
      p = m->bptr;
      break;
    default:;
      return set_log(NOMP_USER_KNL_ARG_TYPE_IS_INVALID, NOMP_ERROR,
                     "Kernel argument type %d passed to libnomp is not valid.",
                     type);
      break;
    }
    arg_list[i]=p;
  }
  
  char *nomp_dir=getenv("NOMP_INSTALL_DIR");

  void* handle = dlopen("libkernellib.so.0.0.1", RTLD_LAZY);
  if (!handle) {
    printf("error \n");

  }

  

  size_t global[3];
  for (unsigned i = 0; i < prg->ndim; i++)
    global[i] = prg->global[i] * prg->local[i];
  if (prg->ndim == 1) {
    sycl::range global_range = sycl::range(global[0]);
    sycl::range local_range = sycl::range(prg->local[0]);
    sycl::nd_range<1> nd_range = sycl::nd_range(global_range, local_range);
    typedef void (*kernel_function_1)(sycl::queue , sycl::nd_range<1> , unsigned int , void **);
    kernel_function_1 hello = (kernel_function_1)dlsym(handle, "kernel_function_1");
     if (!hello) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }
    hello(sycl->queue,nd_range,prg->nargs, arg_list);
  }
  else if (prg->ndim == 2)
  {
    sycl::range global_range = sycl::range(global[0],global[1]);
    sycl::range local_range = sycl::range(prg->local[0],prg->local[1]);
    sycl::nd_range<2> nd_range = sycl::nd_range(global_range, local_range);
    typedef void (*kernel_fun_2)(sycl::queue queue, sycl::nd_range<2> nd_range, unsigned int nargs, void **args);
    kernel_fun_2 hello = (kernel_fun_2)dlsym(handle, "kernel_function_2");
    hello(sycl->queue,nd_range,prg->nargs, arg_list);
  }
  else{
    sycl::range global_range = sycl::range(global[0],global[1],global[2]);
    sycl::range local_range = sycl::range(prg->local[0],prg->local[1],prg->local[2]);
    sycl::nd_range<3> nd_range = sycl::nd_range(global_range, local_range);
    typedef void (*kernel_fun_3)(sycl::queue queue, sycl::nd_range<3> nd_range, unsigned int nargs, void **args);
    kernel_fun_3 hello = (kernel_fun_3)dlsym(handle, "kernel_function_3");
    hello(sycl->queue,nd_range,prg->nargs, arg_list);
  }
  dlclose(handle);
  // FIXME: Wrong. Call set_log()
  return err;
}

static int sycl_finalize(struct backend *bnd) {
   struct sycl_backend *sycl = (sycl_backend *)bnd->bptr;
  // cl_int err = clReleaseCommandQueue(ocl->queue);
  // if (err != CL_SUCCESS)
  //   return set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,
  //                  "command queue release", err);
  // err = clReleaseContext(ocl->ctx);
  // if (err != CL_SUCCESS)
  //   return set_log(NOMP_OPENCL_FAILURE, NOMP_ERROR, ERR_STR_OPENCL_FAILURE,
  //                  "context release", err);
   tfree(bnd->bptr), bnd->bptr = NULL;

  return 0;
}

int sycl_init(struct backend *bnd, const int platform_id, const int device_id) {
  bnd->bptr = tcalloc(struct sycl_backend, 1);
  struct sycl_backend *sycl = (sycl_backend *)bnd->bptr;

  sycl::platform sycl_platform = sycl::platform();
  auto sycl_pplatforms = sycl_platform.get_platforms();

  if (platform_id < 0 | platform_id >= sycl_pplatforms.size())
    return set_log(NOMP_USER_PLATFORM_IS_INVALID, NOMP_ERROR,
                   "Platform id %d provided to libnomp is not valid.",
                   platform_id);
  sycl_platform = sycl_pplatforms[platform_id];
  auto sycl_pdevices = sycl_platform.get_devices();

  if (device_id < 0 || device_id >= sycl_pdevices.size())
    return set_log(NOMP_USER_DEVICE_IS_INVALID, NOMP_ERROR,
                   ERR_STR_USER_DEVICE_IS_INVALID, device_id);

  sycl::device sycl_device = sycl_pdevices[device_id];
  sycl->device_id = sycl_device;
  sycl::context sycl_ctx = sycl::context(sycl_device);

  sycl->ctx = sycl_ctx;
  sycl::queue sycl_queue = sycl::queue(sycl_ctx, sycl_device);
  // clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  sycl->queue = sycl_queue;

  bnd->update = sycl_update;
  bnd->knl_build = sycl_knl_build;
  bnd->knl_run = sycl_knl_run;
  bnd->knl_free = sycl_knl_free;
  bnd->finalize = sycl_finalize;
  return 0;
}
