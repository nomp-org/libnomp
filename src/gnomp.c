#include <gnomp-impl.h>

static void check_handle_(int handle, int max, const char *file, int line) {
  if (handle < 0 || handle >= max) {
    fprintf(stderr, "check_handle failure in %s:%d\n", file, line);
    exit(1);
  }
}

#define check_handle(handle, max) check_handle_(handle, max, __FILE__, __LINE__)

static struct backend *backends = NULL;
static int backends_n = 0;
static int backends_max = 0;

int gnomp_init(int *handle, const char *backend, const int platform,
               const int device) {
  size_t n = strnlen(backend, 32);
  if (n == 32)
    return GNOMP_INVALID_BACKEND;

  char be[BUFSIZ];
  int i;
  for (i = 0; i < n; i++)
    be[i] = tolower(backend[i]);
  be[n] = '\0';

  if (backends_n == backends_max) {
    backends_max += backends_max / 2 + 1;
    backends = (struct backend *)realloc(backends,
                                         sizeof(struct backend) * backends_max);
  }

  int err;
  if (strncmp(be, "opencl", 32) == 0)
    err = opencl_init(&backends[backends_n], platform, device);
  else
    return GNOMP_INVALID_BACKEND;

  if (err == 0)
    *handle = backends_n++;
  else
    *handle = -1;

  return err;
}

static struct mem *mems = NULL;
static int mems_n = 0;
static int mems_max = 0;

static int idx_if_mapped(void *p) {
  int i;
  for (i = 0; i < mems_n; i++)
    if (mems[i].hptr == p)
      break;
  return i;
}

int gnomp_map(void *ptr, const size_t idx0, const size_t idx1,
              const size_t usize, const int direction, const int handle) {
  // See if we mapped this ptr already
  int i = idx_if_mapped(ptr);
  int alloc = 0;
  if (direction == GNOMP_H2D && i == mems_n)
    alloc = 1;

  if (mems_n == mems_max) {
    mems_max += mems_max / 2 + 1;
    mems = (struct mem *)realloc(mems, sizeof(struct mem) * mems_max);
  }

  check_handle(handle, backends_n);
  int err = 0;
  if (backends[handle].backend == GNOMP_OCL)
    err = opencl_map(&backends[handle], &mems[i], ptr, idx0, idx1, usize,
                     direction, alloc);
  else
    err = GNOMP_INVALID_BACKEND;

  if (err == 0 && alloc == 1)
    mems_n++;

  return err;
}

static struct prog *progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

int gnomp_run(int *id, const char *source, const char *name, const int handle,
              const int ndim, const size_t *global, const size_t *local,
              const int nargs, ...) {
  if (progs_n == progs_max) {
    progs_max += progs_max / 2 + 1;
    progs = (struct prog *)realloc(progs, sizeof(struct prog) * progs_max);
  }

  check_handle(handle, backends_n);
  int err = 0;
  if (*id == -1) {
    if (backends[handle].backend == GNOMP_OCL)
      err = opencl_build_knl(&backends[handle], &progs[progs_n], source, name);
    if (err == 0)
      *id = progs_n++;
    else
      return err;
  }

  if (*id >= 0) {
    va_list args;
    va_start(args, nargs);

    int i;
    for (i = 0; i < nargs; i++) {
      /* short, int, long, double, float or pointer */
      int type = va_arg(args, int);
      size_t size, idx;
      union gnomp_arg arg;
      switch (type) {
      case GNOMP_SHORT:
        arg.s = va_arg(args, int);
        size = sizeof(short);
        break;
      case GNOMP_USHORT:
        arg.us = va_arg(args, unsigned int);
        size = sizeof(unsigned short);
        break;
      case GNOMP_INT:
        arg.i = va_arg(args, int);
        size = sizeof(int);
        break;
      case GNOMP_UINT:
        arg.ui = va_arg(args, unsigned int);
        size = sizeof(unsigned int);
        break;
      case GNOMP_LONG:
        arg.l = va_arg(args, long);
        size = sizeof(long);
        break;
      case GNOMP_ULONG:
        arg.ul = va_arg(args, unsigned long);
        size = sizeof(unsigned long);
        break;
      case GNOMP_FLOAT:
        arg.f = va_arg(args, double);
        size = sizeof(float);
        break;
      case GNOMP_DOUBLE:
        arg.d = va_arg(args, double);
        size = sizeof(double);
        break;
      case GNOMP_PTR:
        idx = idx_if_mapped(va_arg(args, void *));
        if (idx < mems_n) {
          arg.p = mems[idx].dptr;
          size = sizeof(mems[idx].dptr);
        } else
          return GNOMP_INVALID_MAP_PTR;
        break;
      default:
        return GNOMP_INVALID_TYPE;
        break;
      }

      if (backends[handle].backend == GNOMP_OCL)
        opencl_set_knl_arg(&progs[*id], i, size, &arg);
    }

    va_end(args);

    if (backends[handle].backend == GNOMP_OCL)
      err = opencl_run_knl(&backends[handle], &progs[*id], ndim, global, local);
    else
      return GNOMP_INVALID_BACKEND;
  }

  return err;
}

int gnomp_err_str(int err_id, char *buf, int buf_size) {
  switch (err_id) {
  case GNOMP_INVALID_BACKEND:
    strncpy(buf, "Invalid gnomp backend", buf_size);
    break;
  case GNOMP_INVALID_PLATFORM:
    strncpy(buf, "Invalid gnomp platform", buf_size);
    break;
  case GNOMP_INVALID_DEVICE:
    strncpy(buf, "Invalid gnomp device", buf_size);
    break;
  case GNOMP_INVALID_TYPE:
    strncpy(buf, "Invalid gnomp type", buf_size);
    break;
  case GNOMP_INVALID_MAP_PTR:
    strncpy(buf, "Invalid gnomp map pointer", buf_size);
    break;
  case GNOMP_MALLOC_ERROR:
    strncpy(buf, "gnomp malloc error", buf_size);
    break;
  default:
    break;
  }

  return 0;
}

int gnomp_finalize(int *handle) {
  check_handle(*handle, backends_n);

  int err = 0;
  if (backends[*handle].backend == GNOMP_OCL)
    err = opencl_finalize(&backends[*handle]);
  else
    return GNOMP_INVALID_BACKEND;

  if (err == 0)
    *handle = -1;
  return err;
}

#undef check_handle
