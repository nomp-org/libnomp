#include <gnomp-impl.h>

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
  } else if (strncmp(be, "cuda", 32) == 0) {
    // err = cuda_init(device);
  } else {
    return GNOMP_INVALID_BACKEND;
  }

  backends_n++;

  return 0;
}

static struct mem *mems = NULL;
static int mems_n = 0;
static int mems_max = 0;

int gnomp_map_to(void *ptr, size_t id0, size_t id1, size_t usize, int handle) {
  // See if we mapped this ptr already
  int i;
  for (i = 0; i < mems_n; i++)
    if (mems[i].h_ptr == ptr)
      break;

  if (mems_n == mems_max) {
    mems_max += mems_max / 2 + 1;
    mems = (struct mem *)realloc(mems, sizeof(struct mem) * mems_max);
  }

  int alloc = i == mems_n ? 1 : 0;
  if (backends[handle].backend == GNOMP_OCL)
    return opencl_map_to(&backends[handle], &mems[i], ptr, id0, id1, usize,
                         alloc);

  if (alloc)
    mems_n++;

  return 0;
}

int gnomp_map_from(void *ptr, size_t id0, size_t id1, size_t usize,
                   int handle) {
  // See if we mapped this ptr already
  int i;
  for (i = 0; i < mems_n; i++)
    if (mems[i].h_ptr == ptr)
      break;

  if (i == mems_n)
    return GNOMP_INVALID_MAP_PTR;

  if (backends[handle].backend == GNOMP_OCL)
    return opencl_map_from(&backends[handle], &mems[i], id0, id1, usize);

  return 0;
}
