#include <nomp-impl.h>

#define BESIZE 1024

#define check_handle(handle, max)                                              \
  do {                                                                         \
    if (handle < 0 || handle >= max)                                           \
      return NOMP_INVALID_HANDLE;                                              \
  } while (0)

static struct backend *backends = NULL;
static int backends_n = 0;
static int backends_max = 0;

int nomp_init(int *handle, const char *backend, const int platform,
              const int device) {
  size_t n = strnlen(backend, BESIZE);
  if (n == BESIZE)
    return NOMP_INVALID_BACKEND;

  char be[BESIZE];
  int i;
  for (i = 0; i < n; i++)
    be[i] = tolower(backend[i]);
  be[n] = '\0';

  if (backends_n == backends_max) {
    backends_max += backends_max / 2 + 1;
    backends = (struct backend *)realloc(backends,
                                         sizeof(struct backend) * backends_max);
  }

  int err = NOMP_INVALID_BACKEND;
  if (strncmp(be, "opencl", 32) == 0)
    err = opencl_init(&backends[backends_n], platform, device);

  *handle = err == 0 ? backends_n++ : -1;

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
    if (mems[i].hptr != NULL && mems[i].hptr == p)
      break;
  return i;
}

int nomp_map(void *ptr, const size_t idx0, const size_t idx1,
             const size_t usize, const int op_, const int handle) {
  check_handle(handle, backends_n);

  if (mems_n == mems_max) {
    mems_max += mems_max / 2 + 1;
    mems = (struct mem *)realloc(mems, sizeof(struct mem) * mems_max);
  }

  int op = op_;

  int idx = idx_if_mapped(ptr);
  if (idx == mems_n) {
    if (op == NOMP_D2H || op == NOMP_FREE)
      return NOMP_INVALID_MAP_PTR;
    else if (op == NOMP_H2D)
      op |= NOMP_ALLOC;

    mems[idx].idx0 = idx0;
    mems[idx].idx1 = idx1;
    mems[idx].usize = usize;
    mems[idx].hptr = ptr;
  }

  int err = NOMP_INVALID_BACKEND;
  if (backends[handle].backend == NOMP_OCL)
    err = opencl_map(&backends[handle], &mems[idx], op);

  if (err == 0) {
    if (idx == mems_n)
      mems_n++;
    else if (op == NOMP_FREE) {
      mems[idx].hptr = NULL;
      mems_n--;
      if (mems_n == 0)
        free(mems);
    }
  }

  return err;
}

static struct prog *progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

static int get_mem_ptr(union nomp_arg *arg, size_t *size, int handle,
                       void *ptr) {
  int err = NOMP_INVALID_MAP_PTR;
  int idx = idx_if_mapped(ptr);
  if (idx < mems_n) {
    err = NOMP_INVALID_BACKEND;
    if (backends[handle].backend == NOMP_OCL)
      err = opencl_get_mem_ptr(arg, size, &mems[idx]);
  }

  return err;
}

int nomp_run(int *id, const char *source, const char *name, const int handle,
             const int ndim, const size_t *global, const size_t *local,
             const int nargs, ...) {
  check_handle(handle, backends_n);

  if (progs_n == progs_max) {
    progs_max += progs_max / 2 + 1;
    progs = (struct prog *)realloc(progs, sizeof(struct prog) * progs_max);
  }

  int err = 0;
  if (*id == -1) {
    err = NOMP_INVALID_BACKEND;
    if (backends[handle].backend == NOMP_OCL)
      err = opencl_build_knl(&backends[handle], &progs[progs_n], source, name);

    if (err == 0)
      *id = progs_n++;
  }

  if (*id >= 0) { // if id < 0, then there is an error
    va_list args;
    va_start(args, nargs);

    int i;
    for (i = 0; i < nargs; i++) {
      int type = va_arg(args, int);
      size_t size;
      union nomp_arg arg;
      switch (type) {
      case NOMP_SHORT:
        arg.s = va_arg(args, int);
        size = sizeof(short);
        break;
      case NOMP_USHORT:
        arg.us = va_arg(args, unsigned int);
        size = sizeof(unsigned short);
        break;
      case NOMP_INT:
        arg.i = va_arg(args, int);
        size = sizeof(int);
        break;
      case NOMP_UINT:
        arg.ui = va_arg(args, unsigned int);
        size = sizeof(unsigned int);
        break;
      case NOMP_LONG:
        arg.l = va_arg(args, long);
        size = sizeof(long);
        break;
      case NOMP_ULONG:
        arg.ul = va_arg(args, unsigned long);
        size = sizeof(unsigned long);
        break;
      case NOMP_FLOAT:
        arg.f = va_arg(args, double);
        size = sizeof(float);
        break;
      case NOMP_DOUBLE:
        arg.d = va_arg(args, double);
        size = sizeof(double);
        break;
      case NOMP_PTR:
        err = get_mem_ptr(&arg, &size, handle, va_arg(args, void *));
        break;
      default:
        err = NOMP_INVALID_TYPE;
        break;
      }

      if (err == 0) {
        err = NOMP_INVALID_BACKEND;
        if (backends[handle].backend == NOMP_OCL)
          err = opencl_set_knl_arg(&progs[*id], i, size, &arg);
      } else
        break;
    }

    va_end(args);

    if (err == 0) {
      err = NOMP_INVALID_BACKEND;
      if (backends[handle].backend == NOMP_OCL)
        err =
            opencl_run_knl(&backends[handle], &progs[*id], ndim, global, local);
    }
  }

  return err;
}

int nomp_err_str(int err_id, char *buf, int buf_size) {
  switch (err_id) {
  case NOMP_INVALID_BACKEND:
    strncpy(buf, "Invalid nomp backend", buf_size);
    break;
  case NOMP_INVALID_PLATFORM:
    strncpy(buf, "Invalid nomp platform", buf_size);
    break;
  case NOMP_INVALID_DEVICE:
    strncpy(buf, "Invalid nomp device", buf_size);
    break;
  case NOMP_INVALID_TYPE:
    strncpy(buf, "Invalid nomp type", buf_size);
    break;
  case NOMP_INVALID_MAP_PTR:
    strncpy(buf, "Invalid nomp map pointer", buf_size);
    break;
  case NOMP_MALLOC_ERROR:
    strncpy(buf, "nomp malloc error", buf_size);
    break;
  default:
    return NOMP_INVALID_ERROR;
    break;
  }

  return 0;
}

int nomp_finalize(int *handle) {
  check_handle(*handle, backends_n);

  int err = NOMP_INVALID_BACKEND;
  if (backends[*handle].backend == NOMP_OCL)
    err = opencl_finalize(&backends[*handle]);

  if (err == 0) {
    backends_n--;
    *handle = -1;
    if (backends_n == 0)
      free(backends);
  }
  return err;
}

#undef check_handle
#undef BESIZE
