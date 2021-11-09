#include <nomp-impl.h>

static int check_handle(int handle, int max) {
  if (handle < 0 || handle >= max)
    return 1;
  else
    return 0;
}

static struct backend *backends = NULL;
static int backends_n = 0;
static int backends_max = 0;

int nomp_init(int *handle, const char *backend, const int platform,
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

/// Returns the pointer to the allocated memory corresponding to 'p'.
/// If no buffer has been allocated for 'p' returns *mems_n*.
static int idx_if_mapped(void *p) {
  // FIXME: This is O(N) in number of allocations.
  // Needs to go. Must store a hashmap.
  int i;
  for (i = 0; i < mems_n; i++)
    if (mems[i].hptr == p)
      break;
  return i;
}

int nomp_map(void *ptr, const size_t idx0, const size_t idx1,
             const size_t usize, const int op_, const int handle) {
  if (check_handle(handle, backends_n) != 0)
    return GNOMP_INVALID_HANDLE;

  if (mems_n == mems_max) {
    mems_max += mems_max / 2 + 1;
    mems = (struct mem *)realloc(mems, sizeof(struct mem) * mems_max);
  }

  int idx = idx_if_mapped(ptr);
  int op = op_;
  if (idx == mems_n) {
    if (op == GNOMP_D2H)
      return GNOMP_INVALID_MAP_PTR;
    else if (op == GNOMP_H2D)
      op |= GNOMP_ALLOC;

    mems[idx].idx0 = idx0;
    mems[idx].idx1 = idx1;
    mems[idx].usize = usize;
    mems[idx].hptr = ptr;
  }

  int err = 0;
  if (backends[handle].backend == GNOMP_OCL)
    err = opencl_map(&backends[handle], &mems[idx], op);
  else
    err = GNOMP_INVALID_BACKEND;

  if (err == 0 && idx == mems_n)
    ++mems_n;

  return err;
}

static struct prog *progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

static int get_mem_ptr(union nomp_arg *arg, size_t *size, int handle,
                       void *ptr) {
  unsigned int idx = idx_if_mapped(ptr);
  int err = GNOMP_INVALID_MAP_PTR;
  if (idx < mems_n) {
    if (backends[handle].backend == GNOMP_OCL)
      err = opencl_get_mem_ptr(arg, size, &mems[idx]);
    else
      err = GNOMP_INVALID_BACKEND;
  }

  return err;
}

int nomp_run(int *id, const char *source, const char *name, const int handle,
             const int ndim, const size_t *global, const size_t *local,
             const int nargs, ...) {
  if (check_handle(handle, backends_n) != 0)
    return GNOMP_INVALID_HANDLE;

  if (progs_n == progs_max) {
    progs_max += progs_max / 2 + 1;
    progs = (struct prog *)realloc(progs, sizeof(struct prog) * progs_max);
  }

  int err = 0;
  if (*id == -1) {
    if (backends[handle].backend == GNOMP_OCL)
      err = opencl_build_knl(&backends[handle], &progs[progs_n], source, name);
    else
      err = GNOMP_INVALID_BACKEND;

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
      size_t size;
      union nomp_arg arg;
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
        err = get_mem_ptr(&arg, &size, handle, va_arg(args, void *));
        break;
      default:
        err = GNOMP_INVALID_TYPE;
        break;
      }

      if (err != 0)
        return err;

      if (backends[handle].backend == GNOMP_OCL) {
        err = opencl_set_knl_arg(&progs[*id], i, size, &arg);
      } else
        err = GNOMP_INVALID_BACKEND;

      if (err != 0)
        return err;
    }

    va_end(args);

    if (backends[handle].backend == GNOMP_OCL)
      err = opencl_run_knl(&backends[handle], &progs[*id], ndim, global, local);
    else
      err = GNOMP_INVALID_BACKEND;
  }

  return err;
}

int nomp_err_str(int err_id, char *buf, int buf_size) {
  switch (err_id) {
  case GNOMP_INVALID_BACKEND:
    strncpy(buf, "Invalid nomp backend", buf_size);
    break;
  case GNOMP_INVALID_PLATFORM:
    strncpy(buf, "Invalid nomp platform", buf_size);
    break;
  case GNOMP_INVALID_DEVICE:
    strncpy(buf, "Invalid nomp device", buf_size);
    break;
  case GNOMP_INVALID_TYPE:
    strncpy(buf, "Invalid nomp type", buf_size);
    break;
  case GNOMP_INVALID_MAP_PTR:
    strncpy(buf, "Invalid nomp map pointer", buf_size);
    break;
  case GNOMP_MALLOC_ERROR:
    strncpy(buf, "nomp malloc error", buf_size);
    break;
  default:
    return GNOMP_INVALID_ERROR;
    break;
  }

  return 0;
}

int nomp_finalize(int *handle) {
  if (check_handle(*handle, backends_n) != 0)
    return GNOMP_INVALID_HANDLE;

  int err = 0;
  if (backends[*handle].backend == GNOMP_OCL)
    err = opencl_finalize(&backends[*handle]);
  else
    err = GNOMP_INVALID_BACKEND;

  if (err == 0) {
    backends_n--;
    *handle = -1;
    if (backends_n == 0)
      free(backends);
  }
  return err;
}

#undef check_handle
