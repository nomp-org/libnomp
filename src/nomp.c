#include <nomp-impl.h>
#include <pthread.h>

#define BESIZE 1024

static struct backend nomp;
static int initialized = 0;
static pthread_mutex_t m;

//=============================================================================
// nomp_init
//
int nomp_init(const char *backend, const int platform, const int device) {
  pthread_mutex_lock(&m);

  if (initialized > 0) {
    pthread_mutex_unlock(&m);
    return NOMP_INITIALIZED_ERROR;
  }

  char be[BESIZE];
  size_t n = strnlen(backend, BESIZE);
  int i;
  for (i = 0; i < n; i++)
    be[i] = tolower(backend[i]);
  be[n] = '\0';

  int err = 0;
  if (strncmp(be, "opencl", 32) == 0)
    err = opencl_init(&nomp, platform, device);
  else
    err = NOMP_INVALID_BACKEND;

  if (err == 0)
    initialized = 1;

  pthread_mutex_unlock(&m);

  return err;
}

//=============================================================================
// nomp_map
//
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

  int err = nomp.map(&nomp, &mems[idx], op);
  if (err == 0) {
    if (idx == mems_n)
      mems_n++;
    else if (op == NOMP_FREE)
      mems[idx].hptr = NULL;
  }

  return err;
}

//=============================================================================
// nomp_run
//
static struct prog *progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

int nomp_run(int *id, const char *source, const char *name, const int ndim,
             const size_t *global, const size_t *local, const int nargs, ...) {
  if (progs_n == progs_max) {
    progs_max += progs_max / 2 + 1;
    progs = (struct prog *)realloc(progs, sizeof(struct prog) * progs_max);
  }

  if (*id == -1) {
    if (nomp.knl_build(&nomp, &progs[progs_n], source, name) == 0)
      *id = progs_n++;
    else
      return NOMP_KNL_BUILD_ERROR;
  }

  if (*id >= 0) { // if id < 0, then there is an error
    va_list args;
    va_start(args, nargs);

    int i, idx;
    for (i = 0; i < nargs; i++) {
      int type = va_arg(args, int);
      void *p = va_arg(args, void *);
      size_t size;
      switch (type) {
      case NOMP_SCALAR:
        size = va_arg(args, size_t);
        break;
      case NOMP_PTR:
        if ((idx = idx_if_mapped(p)) < mems_n)
          nomp.map_ptr(&p, &size, &mems[idx]);
        else
          return NOMP_INVALID_MAP_PTR;
        break;
      default:
        return NOMP_KNL_ARG_TYPE_ERROR;
        break;
      }

      if (nomp.knl_set(&progs[*id], i, size, p) != 0)
        return NOMP_KNL_ARG_SET_ERROR;
    }

    va_end(args);

    if (nomp.knl_run(&nomp, &progs[*id], ndim, global, local) != 0)
      return NOMP_KNL_RUN_ERROR;
  }

  return 0;
}

//=============================================================================
// nomp_err
//
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
    strncpy(buf, "Nomp malloc error", buf_size);
    break;
  case NOMP_INVALID_MAP_OP:
    strncpy(buf, "Invalid map operation", buf_size);
    break;
  case NOMP_INITIALIZED_ERROR:
    strncpy(buf, "Nomp is already initialized", buf_size);
    break;
  case NOMP_NOT_INITIALIZED_ERROR:
    strncpy(buf, "Nomp is not initialized", buf_size);
    break;
  case NOMP_KNL_BUILD_ERROR:
    strncpy(buf, "Nomp kernel build failed", buf_size);
    break;
  case NOMP_KNL_ARG_TYPE_ERROR:
    strncpy(buf, "Invalid nomp kernel argument type", buf_size);
    break;
  case NOMP_KNL_ARG_SET_ERROR:
    strncpy(buf, "Nomp kernel argument set failed", buf_size);
    break;
  case NOMP_KNL_RUN_ERROR:
    strncpy(buf, "Nomp kernel run failed", buf_size);
    break;
  default:
    break;
  }

  return 0;
}

//=============================================================================
// nomp_finalize
//
int nomp_finalize(void) {
  pthread_mutex_lock(&m);

  if (!initialized) {
    pthread_mutex_unlock(&m);
    return NOMP_NOT_INITIALIZED_ERROR;
  }

  int i;
  for (i = 0; i < mems_n && nomp.map(&nomp, &mems[i], NOMP_FREE) == 0; i++)
    ;
  if (i == mems_n)
    free(mems);
  else {
    pthread_mutex_unlock(&m);
    return NOMP_INVALID_MAP_PTR;
  }

  for (i = 0; i < progs_n && nomp.knl_free(&progs[i]) == 0; i++)
    ;
  if (i == progs_n)
    free(progs);
  else {
    pthread_mutex_unlock(&m);
    return NOMP_INVALID_KNL;
  }

  if (nomp.finalize(&nomp) == 0)
    initialized = 0;
  else {
    pthread_mutex_unlock(&m);
    return NOMP_FINALIZE_FAILURE;
  }

  pthread_mutex_unlock(&m);
  return 0;
}

#undef BESIZE
