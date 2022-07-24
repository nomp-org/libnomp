#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "nomp-impl.h"
#include <pthread.h>

static struct backend nomp;
static int initialized = 0;
static pthread_mutex_t m;

//=============================================================================
// nomp_init
//
int nomp_init(const char *backend, int platform, int device) {
  pthread_mutex_lock(&m);
  if (initialized) {
    pthread_mutex_unlock(&m);
    return NOMP_INITIALIZED_ERROR;
  }

  char be[BUFSIZ];
  size_t n = strnlen(backend, BUFSIZ);
  for (int i = 0; i < n; i++)
    be[i] = tolower(backend[i]);
  be[n] = '\0';

  int err = NOMP_INVALID_BACKEND;
  if (strncmp(be, "opencl", 32) == 0)
    err = opencl_init(&nomp, platform, device);

  if (!Py_IsInitialized()) {
    Py_Initialize();
    // Append current working dir
    py_append_to_sys_path(".");
    // There should be a better way to figure the installation
    // path based on the shared library path
    err = NOMP_INSTALL_DIR_NOT_FOUND;
    char *val = getenv("NOMP_INSTALL_DIR");
    if (val) {
      const char *python_dir = "python", *py_module = "c_to_loopy";
      size_t len0 = strlen(val), len1 = strlen(python_dir);
      char *abs_dir = (char *)calloc(len0 + len1 + 2, sizeof(char));
      strncpy(abs_dir, val, len0), strncpy(abs_dir + len0, "/", 1);
      strncpy(abs_dir + len0 + 1, python_dir, len1);
      py_append_to_sys_path(abs_dir);
      free(abs_dir), err = 0;
    }
  }
  if (err)
    return err;

  initialized = !err;
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
  for (int i = 0; i < mems_n; i++)
    if (mems[i].bptr != NULL && mems[i].hptr == p)
      return i;
  return mems_n;
}

int nomp_map(void *ptr, size_t idx0, size_t idx1, size_t usize, int op) {
  int idx = idx_if_mapped(ptr);
  if (idx == mems_n || mems[idx].bptr == NULL) {
    if (op == NOMP_D2H || op == NOMP_FREE)
      return NOMP_INVALID_MAP_PTR;
    op |= NOMP_ALLOC;
  }

  if (idx == mems_n) {
    if (mems_n == mems_max) {
      mems_max += mems_max / 2 + 1;
      mems = (struct mem *)realloc(mems, sizeof(struct mem) * mems_max);
    }
    mems[idx].idx0 = idx0, mems[idx].idx1 = idx1, mems[idx].usize = usize;
    mems[idx].hptr = ptr, mems[idx].bptr = NULL;
  }

  if ((op & NOMP_ALLOC) && mems[idx].bptr != NULL)
    return NOMP_INVALID_MAP_PTR;

  if (mems[idx].idx0 != idx0 || mems[idx].idx1 != idx1 ||
      mems[idx].usize != usize)
    return NOMP_INVALID_MAP_PTR;

  int err = nomp.map(&nomp, &mems[idx], op);
  mems_n += (idx == mems_n) && (err == 0);

  return err;
}

//=============================================================================
// nomp_jit
//
static struct prog *progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

int nomp_jit(int *id, int *ndim, size_t *global, size_t *local,
             const char *c_src, const char *annotations, const char *callback) {
  if (*id == -1) {
    if (progs_n == progs_max) {
      progs_max += progs_max / 2 + 1;
      progs = (struct prog *)realloc(progs, sizeof(struct prog) * progs_max);
    }

    size_t len = strlen(callback) + 1;
    char *callback_ = (char *)calloc(len, sizeof(char));
    strncpy(callback_, callback, len);

    const char colon[2] = ":";
    char *py_file = strtok(callback_, colon), *py_func = NULL;
    if (py_file != NULL)
      py_func = strtok(NULL, colon);

    struct knl knl = {.src = NULL,
                      .name = NULL,
                      .ndim = 0,
                      .gsize = {0, 0, 0},
                      .lsize = {0, 0, 0}};
    int err = py_user_callback(&knl, c_src, py_file, py_func);
    free(callback_);
    if (err)
      return err;

    for (int i = 0; i < knl.ndim; i++) {
      global[i] = knl.gsize[i];
      local[i] = knl.lsize[i];
    }
    *ndim = knl.ndim;

    if (nomp.knl_build(&nomp, &progs[progs_n], knl.src, knl.name) == 0)
      *id = progs_n++;
    else
      return NOMP_KNL_BUILD_ERROR;
  }

  return 0;
}

//=============================================================================
// nomp_run
//
int nomp_run(int id, int ndim, const size_t *global, const size_t *local,
             int nargs, ...) {
  if (id >= 0) {
    va_list args;
    va_start(args, nargs);
    for (int i = 0; i < nargs; i++) {
      int type = va_arg(args, int), idx;
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

      if (nomp.knl_set(&progs[id], i, size, p) != 0)
        return NOMP_KNL_ARG_SET_ERROR;
    }
    va_end(args);

    if (nomp.knl_run(&nomp, &progs[id], ndim, global, local) != 0)
      return NOMP_KNL_RUN_ERROR;
    return 0;
  }
  return NOMP_INVALID_KNL;
}

//=============================================================================
// nomp_err
//
int nomp_err(char *buf, int err, size_t buf_size) {
  switch (err) {
  case NOMP_INVALID_BACKEND:
    strncpy(buf, "Invalid NOMP backend", buf_size);
    break;
  case NOMP_INVALID_PLATFORM:
    strncpy(buf, "Invalid NOMP platform", buf_size);
    break;
  case NOMP_INVALID_DEVICE:
    strncpy(buf, "Invalid NOMP device", buf_size);
    break;
  case NOMP_INVALID_MAP_PTR:
    strncpy(buf, "Invalid NOMP map pointer", buf_size);
    break;
  case NOMP_INVALID_MAP_OP:
    strncpy(buf, "Invalid map operation", buf_size);
    break;
  case NOMP_INVALID_KNL:
    strncpy(buf, "Invalid NOMP kernel", buf_size);
    break;

  case NOMP_INITIALIZED_ERROR:
    strncpy(buf, "NOMP is already initialized", buf_size);
    break;
  case NOMP_NOT_INITIALIZED_ERROR:
    strncpy(buf, "NOMP is not initialized", buf_size);
    break;
  case NOMP_FINALIZE_ERROR:
    strncpy(buf, "Failed to finalize NOMP", buf_size);
    break;
  case NOMP_MALLOC_ERROR:
    strncpy(buf, "NOMP malloc error", buf_size);
    break;

  case NOMP_INSTALL_DIR_NOT_FOUND:
    strncpy(buf, "NOMP_INSTALL_DIR env. variable is not set", buf_size);
    break;
  case NOMP_USER_CALLBACK_NOT_FOUND:
    strncpy(buf, "Specified user callback function not found", buf_size);
    break;
  case NOMP_USER_CALLBACK_FAILURE:
    strncpy(buf, "User callback function failed", buf_size);
    break;
  case NOMP_C_TO_LOOPY_CONVERSION_ERROR:
    strncpy(buf, "C to Loopy conversion failed", buf_size);
    break;
  case NOMP_CODEGEN_FAILED:
    strncpy(buf, "Code generation from loopy kernel failed", buf_size);
    break;

  case NOMP_KNL_BUILD_ERROR:
    strncpy(buf, "NOMP kernel build failed", buf_size);
    break;
  case NOMP_KNL_ARG_TYPE_ERROR:
    strncpy(buf, "Invalid NOMP kernel argument type", buf_size);
    break;
  case NOMP_KNL_ARG_SET_ERROR:
    strncpy(buf, "Setting NOMP kernel argument failed", buf_size);
    break;
  case NOMP_KNL_RUN_ERROR:
    strncpy(buf, "NOMP kernel run failed", buf_size);
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

  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i].bptr != NULL)
      nomp.map(&nomp, &mems[i], NOMP_FREE);
  }
  free(mems), mems = NULL, mems_n = mems_max = 0;

  for (unsigned i = 0; i < progs_n; i++) {
    if (progs[i].bptr != NULL)
      nomp.knl_free(&progs[i]);
  }
  free(progs), progs = NULL, progs_n = progs_max = 0;

  if (nomp.finalize(&nomp) == 0)
    initialized = 0;
  else {
    pthread_mutex_unlock(&m);
    return NOMP_FINALIZE_ERROR;
  }

  pthread_mutex_unlock(&m);
  return 0;
}

//=============================================================================
// Helper functions: nomp_err & nomp_assert
//
void nomp_chk_(int err, const char *file, unsigned line) {
  if (err) {
    char buf[2 * BUFSIZ];
    nomp_err(buf, err, 2 * BUFSIZ);
    printf("%s:%d %s\n", file, line, buf);
    exit(1);
  }
}

void nomp_assert_(int cond, const char *file, unsigned line) {
  if (!cond) {
    printf("nomp_assert failure at %s:%d\n", file, line);
    exit(1);
  }
}
