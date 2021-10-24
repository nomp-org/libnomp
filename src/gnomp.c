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

int gnomp_init(char *backend, int platform, int device) {
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
  if (strncmp(be, "opencl", 32) == 0) {
    err = opencl_init(&backends[backends_n], platform, device);
  } else {
    return GNOMP_INVALID_BACKEND;
  }

  if (err == 0)
    backends_n++;

  return err;
}

static struct mem *mems = NULL;
static int mems_n = 0;
static int mems_max = 0;

int gnomp_map(void *ptr, size_t idx0, size_t idx1, size_t usize, int direction,
              int handle) {
  // See if we mapped this ptr already
  int i;
  for (i = 0; i < mems_n; i++)
    if (mems[i].h_ptr == ptr)
      break;

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

int gnomp_build_program(const char *source, int handle) {
  if (progs_n == progs_max) {
    progs_max += progs_max / 2 + 1;
    progs = (struct prog *)realloc(progs, sizeof(struct prog) * progs_max);
  }

  check_handle(handle, backends_n);
  int err;
  if (backends[handle].backend == GNOMP_OCL)
    err = opencl_build_program(&backends[handle], &progs[progs_n], source);

  if (err == 0)
    progs_n++;

  return err;
}

#undef check_handle
