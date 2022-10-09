#include "nomp-impl.h"
#include "nomp-log-str.h"

static struct backend nomp;
static int initialized = 0;

//=============================================================================
// nomp_init
//
int nomp_init(const char *backend, int platform, int device) {
  if (initialized)
    return nomp_set_log(NOMP_INITIALIZED_ERROR, NOMP_ERROR,
                        ERR_STR_NOMP_IS_ALREADY_INITIALIZED, nomp.name);

  char name[MAX_BACKEND_NAME_SIZE];
  size_t n = strnlen(backend, MAX_BACKEND_NAME_SIZE);
  for (int i = 0; i < n; i++)
    name[i] = tolower(backend[i]);
  name[n] = '\0';

  int err = 0;
  if (strncmp(name, "opencl", MAX_BACKEND_NAME_SIZE) == 0) {
#if defined(OPENCL_ENABLED)
    err = opencl_init(&nomp, platform, device);
#endif
  } else if (strncmp(name, "cuda", MAX_BACKEND_NAME_SIZE) == 0) {
#if defined(CUDA_ENABLED)
    err = cuda_init(&nomp, platform, device);
#endif
  } else {
    err = nomp_set_log(NOMP_INVALID_BACKEND, NOMP_ERROR,
                       ERR_STR_FAILED_TO_INITIALIZE_NOMP, name);
  }
  return_on_err(err);

  strncpy(nomp.name, name, MAX_BACKEND_NAME_SIZE);

  if (!Py_IsInitialized()) {
    // May be we need the isolated configuration listed here:
    // https://docs.python.org/3/c-api/init_config.html#init-config
    // But for now, we do the simplest thing possible.
    Py_Initialize();
    // Append current working dir
    py_append_to_sys_path(".");
    // There should be a better way to figure the installation
    // path based on the shared library path
    char *install_dir = getenv("NOMP_INSTALL_DIR");
    if (install_dir) {
      char *abs_dir = strcatn(3, install_dir, "/", py_dir);
      py_append_to_sys_path(abs_dir);
      FREE(abs_dir);
    } else {
      return nomp_set_log(NOMP_INSTALL_DIR_NOT_FOUND, NOMP_ERROR,
                          ERR_STR_NOMP_INSTALL_DIR_NOT_SET);
    }
  } else {
    // TODO: Check if we can use initialized python
    err = nomp_set_log(NOMP_PY_INITIALIZE_ERROR, NOMP_WARNING,
                       WARNING_STR_PYTHON_IS_ALREADY_INITIALIZED);
  }

  initialized = 1;
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
struct mem *mem_if_mapped(void *p) {
  // FIXME: This is O(N) in number of allocations.
  // Needs to go. Must store a hashmap.
  for (int i = 0; i < mems_n; i++)
    if (mems[i].bptr != NULL && mems[i].hptr == p)
      return &mems[i];
  return NULL;
}

int nomp_map(void *ptr, size_t idx0, size_t idx1, size_t usize, int op) {
  struct mem *m = mem_if_mapped(ptr);
  if (m == NULL) {
    if (op == NOMP_D2H || op == NOMP_FREE)
      return nomp_set_log(NOMP_INVALID_MAP_PTR, NOMP_ERROR,
                          ERR_STR_INVALID_MAP_OP, op);
    op |= NOMP_ALLOC;
  }

  if (m == NULL) {
    if (mems_n == mems_max) {
      mems_max += mems_max / 2 + 1;
      mems = (struct mem *)realloc(mems, sizeof(struct mem) * mems_max);
    }
    m = &mems[mems_n], mems_n++;
    m->idx0 = idx0, m->idx1 = idx1, m->usize = usize;
    m->hptr = ptr, m->bptr = NULL;
  }

  if (m->idx0 != idx0 || m->idx1 != idx1 || m->usize != usize)
    return nomp_set_log(NOMP_INVALID_MAP_PTR, NOMP_ERROR,
                        ERR_STR_INVALID_MAP_PTR, ptr);
  if ((op & NOMP_ALLOC) && m->bptr != NULL)
    return nomp_set_log(NOMP_PTR_ALREADY_MAPPED, NOMP_ERROR,
                        ERR_STR_PTR_IS_ALREADY_ALLOCATED, ptr);
  return nomp.map(&nomp, m, op);
}

//=============================================================================
// nomp_jit
//
static struct prog *progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

int nomp_jit(int *id, const char *c_src, const char *annotations,
             const char *callback, unsigned nargs, const char *args, ...) {
  if (*id == -1) {
    if (progs_n == progs_max) {
      progs_max += progs_max / 2 + 1;
      progs = (struct prog *)realloc(progs, sizeof(struct prog) * progs_max);
    }

    // Create loopy kernel from C source
    PyObject *pKnl = NULL;
    int err = py_c_to_loopy(&pKnl, c_src, nomp.name);
    return_on_err(err);

    // Call the User callback function
    char *callback_ = strndup(callback, BUFSIZ),
         *user_file = strtok(callback_, ":"), *user_func = NULL;
    if (user_file)
      user_func = strtok(NULL, ":");
    err = py_user_callback(&pKnl, user_file, user_func);
    FREE(callback_);
    return_on_err(err);

    // Get OpenCL, CUDA, etc. source and name from the loopy kernel
    char *name, *src;
    err = py_get_knl_name_and_src(&name, &src, pKnl);
    return_on_err(err);

    // Build the kernel
    struct prog *prog = &progs[progs_n];
    err = nomp.knl_build(&nomp, prog, src, name);
    FREE(src);
    FREE(name);
    return_on_err(err);

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

    prog->nargs = nargs;
    err = py_get_grid_size(&prog->ndim, prog->global, prog->local, pKnl, pDict);
    FREE(args_);
    Py_DECREF(pDict), Py_XDECREF(pKnl);
    return_on_err(err);

    *id = progs_n++;
  }

  return 0;
}

//=============================================================================
// nomp_run
//
int nomp_run(int id, ...) {
  if (id >= 0) {
    va_list args;
    va_start(args, id);
    int err = nomp.knl_run(&nomp, &progs[id], args);
    va_end(args);
    if (err)
      return nomp_set_log(NOMP_KNL_RUN_ERROR, NOMP_ERROR,
                          ERR_STR_KERNEL_RUN_FAILED, id);
    return 0;
  }
  return nomp_set_log(NOMP_INVALID_KNL, NOMP_ERROR, ERR_STR_INVALID_KERNEL, id);
}

//=============================================================================
// Helper functions: nomp_assert & nomp_err
//
void nomp_assert_(int cond, const char *file, unsigned line) {
  if (!cond) {
    printf("nomp_assert failure at %s:%d\n", file, line);
    exit(1);
  }
}

static struct log *logs = NULL;
static unsigned logs_n = 0;
static unsigned logs_max = 0;
static const char *LOG_TYPE_STRING[] = {"Error", "Warning", "Information"};

int nomp_err_type_to_str(char *buf, int err, size_t buf_size) {
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

int nomp_set_log_(const char *description, int logno, nomp_log_type type,
                  const char *fname, unsigned line_no, ...) {
  if (logs_max <= logs_n) {
    logs_max += logs_max / 2 + 1;
    logs = (struct log *)realloc(logs, sizeof(struct log) * logs_max);
    if (logs == NULL)
      return NOMP_OUT_OF_MEMORY;
  }

  va_list vargs;
  char buf[BUFSIZ];
  va_start(vargs, line_no);
  vsnprintf(buf, BUFSIZ, description, vargs);
  va_end(vargs);

  const char *log_type_string = LOG_TYPE_STRING[type];
  size_t n_desc = strnlen(buf, BUFSIZ);
  size_t n_file = strnlen(fname, BUFSIZ);
  size_t n_log_type = strnlen(log_type_string, BUFSIZ);
  logs[logs_n].description =
      (char *)calloc(n_desc + n_file + n_log_type + 4 + 3, sizeof(char));
  snprintf(logs[logs_n].description, BUFSIZ + n_file + n_log_type + 4 + 3 + 1,
           "%s:%s:%u %s", log_type_string, fname, line_no, buf);
  logs[logs_n].logno = logno;
  logs[logs_n].type = type;
  logs_n += 1;
  return logs_n;
}

int nomp_get_log(char **log_str, int log_id, nomp_log_type type) {
  if (log_id <= 0 && log_id > logs_n) {
    *log_str = NULL;
    return NOMP_INVALID_LOG_ID;
  }
  struct log lg = logs[log_id - 1];
  if (lg.type != type) {
    return NOMP_LOG_TYPE_MISMATCH;
  }
  size_t n_desc = strnlen(lg.description, BUFSIZ) + 1;
  *log_str = (char *)calloc(n_desc, sizeof(char));
  strncpy(*log_str, lg.description, n_desc + 1);
  return 0;
}

int nomp_get_log_no(int log_id) {
  if (log_id <= 0 && log_id > logs_n) {
    return NOMP_INVALID_LOG_ID;
  }
  return logs[log_id - 1].logno;
}

void nomp_chk_(int err_id, const char *file, unsigned line) {
  if (err_id == 0)
    return;
  char *err_str;
  int err = nomp_get_log(&err_str, err_id, NOMP_ERROR);
  if (err != NOMP_INVALID_LOG_ID && err != NOMP_LOG_TYPE_MISMATCH) {
    printf("%s:%d %s\n", file, line, err_str);
    free(err_str);
    exit(1);
  }
}

//=============================================================================
// nomp_finalize
//
int nomp_finalize(void) {
  if (!initialized)
    return nomp_set_log(NOMP_NOT_INITIALIZED_ERROR, NOMP_ERROR,
                        ERR_STR_NOMP_IS_NOT_INITIALIZED);

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
  FREE(progs);
  progs = NULL, progs_n = progs_max = 0;

  for (unsigned i = 0; i < logs_n; i++)
    FREE(logs[i].description);
  FREE(logs);
  logs = NULL, logs_n = logs_max = 0;

  initialized = nomp.finalize(&nomp);
  if (initialized)
    return nomp_set_log(NOMP_FINALIZE_ERROR, NOMP_ERROR,
                        ERR_STR_FAILED_TO_FINALIZE_NOMP);

  if (Py_IsInitialized())
    Py_Finalize();

  return 0;
}
