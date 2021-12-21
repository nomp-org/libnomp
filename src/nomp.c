#include <nomp-impl.h>

#define BESIZE 1024

static struct backend nomp;
static int initialized = 0;

int nomp_init(const char *backend, const int platform, const int device) {
  if (initialized > 0)
    return NOMP_INITIALIZED_ERROR;

  char be[BESIZE];
  size_t n = strnlen(backend, BESIZE);
  int i;
  for (i = 0; i < n; i++)
    be[i] = tolower(backend[i]);
  be[n] = '\0';

  int err = NOMP_INVALID_BACKEND;
  if (strncmp(be, "opencl", 32) == 0)
    err = opencl_init(&nomp, platform, device);

  if (err == 0)
    initialized = 1;

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
             const size_t usize, const int op_) {
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
  if (nomp.backend == NOMP_OCL)
    err = opencl_map(&nomp, &mems[idx], op);

  if (err == 0)
    if (idx == mems_n)
      mems_n++;
    else if (op == NOMP_FREE)
      mems[idx].hptr = NULL;

  return err;
}

static struct prog *progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

static int map_ptr(union nomp_arg *arg, size_t *size, void *ptr) {
  int err = NOMP_INVALID_MAP_PTR;
  int idx = idx_if_mapped(ptr);
  if (idx < mems_n) {
    err = NOMP_INVALID_BACKEND;
    if (nomp.backend == NOMP_OCL)
      err = opencl_map_ptr(arg, size, &mems[idx]);
  }

  return err;
}

int nomp_run(int *id, const char *source, const char *name, const int ndim,
             const size_t *global, const size_t *local, const int nargs, ...) {
  if (progs_n == progs_max) {
    progs_max += progs_max / 2 + 1;
    progs = (struct prog *)realloc(progs, sizeof(struct prog) * progs_max);
  }

  int err = 0;
  if (*id == -1) {
    err = NOMP_INVALID_BACKEND;
    if (nomp.backend == NOMP_OCL)
      err = opencl_knl_build(&nomp, &progs[progs_n], source, name);

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
        err = map_ptr(&arg, &size, va_arg(args, void *));
        break;
      default:
        err = NOMP_INVALID_TYPE;
        break;
      }

      if (err == 0) {
        err = NOMP_INVALID_BACKEND;
        if (nomp.backend == NOMP_OCL)
          err = opencl_knl_set(&progs[*id], i, size, &arg);
      } else
        break;
    }

    va_end(args);

    if (err == 0) {
      err = NOMP_INVALID_BACKEND;
      if (nomp.backend == NOMP_OCL)
        err = opencl_knl_run(&nomp, &progs[*id], ndim, global, local);
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
  case NOMP_INITIALIZED_ERROR:
    strncpy(buf, "Nomp is already initialized", buf_size);
    break;
  case NOMP_NOT_INITIALIZED_ERROR:
    strncpy(buf, "Nomp is not initialized", buf_size);
    break;
  default:
    return NOMP_INVALID_ERROR;
    break;
  }

  return 0;
}

int nomp_finalize(void) {
  if (!initialized)
    return NOMP_NOT_INITIALIZED_ERROR;

  int i, err;
  if (nomp.backend == NOMP_OCL)
    for (i = err = 0; err == 0 && i < mems_n; i++)
      err = opencl_map(&nomp, &mems[i], NOMP_FREE);
  else
    err = NOMP_INVALID_BACKEND;
  if (err == 0)
    free(mems);

  if (nomp.backend == NOMP_OCL)
    for (i = 0; err == 0 && i < progs_n; i++)
      err = opencl_knl_free(&progs[i]);
  else
    err = NOMP_INVALID_BACKEND;
  if (err == 0)
    free(progs);

  if (err == 0)
    if (nomp.backend == NOMP_OCL)
      err = opencl_finalize(&nomp);

  if (err == 0)
    initialized = 0;

  return err;
}

#undef BESIZE
