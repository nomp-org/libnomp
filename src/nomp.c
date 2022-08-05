#include "nomp-impl.h"

static struct backend nomp;
static int initialized = 0;

#define FREE(x)                                                                \
  do {                                                                         \
    if (x)                                                                     \
      free(x);                                                                 \
  } while (0)

//=============================================================================
// nomp_init
//
int nomp_init(const char *backend, int platform, int device) {
  if (initialized)
    return NOMP_INITIALIZED_ERROR;

  char be[BUFSIZ];
  size_t n = strnlen(backend, BUFSIZ);
  for (int i = 0; i < n; i++)
    be[i] = tolower(backend[i]);
  be[n] = '\0';

  int err = NOMP_INVALID_BACKEND;
  if (strncmp(be, "opencl", 32) == 0)
    err = opencl_init(&nomp, platform, device);
  if (err)
    return err;

  err = NOMP_PY_INITIALIZE_ERROR;
  const char *python_dir = "python", *py_module = "c_to_loopy";
  if (!Py_IsInitialized()) {
    // May be we need the isolated configuration listed here:
    // https://docs.python.org/3/c-api/init_config.html#init-config
    // But for now, we do the simplest thing possible.
    Py_Initialize();
    // Append current working dir
    py_append_to_sys_path(".");
    // There should be a better way to figure the installation
    // path based on the shared library path
    err = NOMP_INSTALL_DIR_NOT_FOUND;
    char *install_dir = getenv("NOMP_INSTALL_DIR");
    if (install_dir) {
      char *abs_dir = strcatn(3, install_dir, "/", python_dir);
      py_append_to_sys_path(abs_dir);
      printf("abs_dir = %s\n", abs_dir);
      FREE(abs_dir);
      err = 0;
    }
  }
  if (err)
    return err;

  initialized = 1;
  return 0;
}

//=============================================================================
// nomp_map
//
static struct mem *mems = NULL;
static int mems_n = 0;
static int mems_max = 0;

/// Returns the pointer to the allocated memory corresponding to 'p'.
/// If no buffer has been allocated for 'p' returns *mems_n*.
static size_t idx_if_mapped(void *p) {
  // FIXME: This is O(N) in number of allocations.
  // Needs to go. Must store a hashmap.
  for (int i = 0; i < mems_n; i++)
    if (mems[i].bptr != NULL && mems[i].hptr == p)
      return i;
  return mems_n;
}

int nomp_map(void *ptr, size_t idx0, size_t idx1, size_t usize, int op) {
  size_t idx = idx_if_mapped(ptr);
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
             const char *c_src, const char *annotations, const char *callback,
             int nargs, const char *args, ...) {
  if (*id == -1) {
    if (progs_n == progs_max) {
      progs_max += progs_max / 2 + 1;
      progs = (struct prog *)realloc(progs, sizeof(struct prog) * progs_max);
    }

    // Create loopy kernel from C source
    PyObject *pKnl = NULL;
    int err = py_convert_from_c_to_loopy(&pKnl, c_src);
    if (err)
      return err;

    // Call the User callback function
    char *callback_ = strndup(callback, BUFSIZ),
         *py_file = strtok(callback_, ":"), *py_func = NULL;
    if (py_file)
      py_func = strtok(NULL, ":");
    err = py_user_callback(&pKnl, py_file, py_func);
    FREE(callback_);
    if (err)
      return err;

    // Get OpenCL, CUDA, etc. source and and name from the loopy kernel and
    // then build it
    char *name, *src;
    err = py_get_knl_name_and_src(&name, &src, pKnl);
    if (err)
      return err;
    err = nomp.knl_build(&nomp, &progs[progs_n], src, name);
    FREE(src);
    FREE(name);
    if (err)
      return err;

    // Get grid size of the loopy kernel after transformations. We will create a
    // dictionary with variable name as keys, variable value as value and then
    // pass it to the function
    char *args_ = strndup(args, BUFSIZ), *arg = strtok(args_, ",");
    PyObject *pDict = PyDict_New();
    va_list vargs;
    va_start(vargs, args);
    for (int i = 0; i < nargs; i++) {
      // FIXME: `type` should be able to distinguish between integer types,
      // floating point types, boolean type or a pointer type. We are assuming
      // it is an integer type for now.
      int type = va_arg(vargs, int);
      size_t size = va_arg(vargs, size_t);
      int *p = (int *)va_arg(vargs, void *);
      PyObject *pKey = PyUnicode_FromStringAndSize(arg, strlen(arg));
      PyObject *pValue = PyLong_FromLong(*p);
      PyDict_SetItem(pDict, pKey, pValue);
      arg = strtok(NULL, ",");
    }
    va_end(vargs);

    py_get_grid_size(ndim, global, local, pKnl, pDict);
    FREE(args_);
    Py_DECREF(pDict), Py_XDECREF(pKnl);

    if (err)
      return NOMP_KNL_BUILD_ERROR;
    *id = progs_n++;
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
      int type = va_arg(args, int);
      void *p = va_arg(args, void *);
      size_t size, idx;
      switch (type) {
      case NOMP_INTEGER:
      case NOMP_FLOAT:
        size = va_arg(args, size_t);
        break;
      case NOMP_PTR:
        idx = idx_if_mapped(p);
        if (idx < mems_n)
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
  case NOMP_LOOPY_CONVERSION_ERROR:
    strncpy(buf, "C to Loopy conversion failed", buf_size);
    break;
  case NOMP_LOOPY_CODEGEN_FAILED:
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
  if (!initialized)
    return NOMP_NOT_INITIALIZED_ERROR;

  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i].bptr != NULL)
      nomp.map(&nomp, &mems[i], NOMP_FREE);
  }
  FREE(mems);
  mems = NULL, mems_n = mems_max = 0;

  for (unsigned i = 0; i < progs_n; i++) {
    if (progs[i].bptr != NULL)
      nomp.knl_free(&progs[i]);
  }
  free(progs), progs = NULL, progs_n = progs_max = 0;

  initialized = nomp.finalize(&nomp);
  if (initialized)
    return NOMP_FINALIZE_ERROR;

  if (Py_IsInitialized())
    Py_Finalize();

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
