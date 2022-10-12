#include "nomp-impl.h"

struct env {
  char *install_dir;
  int verbose;
};

static struct backend nomp;
static struct env envs;
static int initialized = 0;

static char *get_if_env(const char *name) {
  const char *temp = getenv(name);
  if (temp) {
    char *tmp_var = (char *)malloc(strnlen(temp, BUFSIZ));
    if (tmp_var != NULL) {
      strncpy(tmp_var, temp, strnlen(temp, BUFSIZ));
      return tmp_var;
    }
  }
  return NULL;
}

void nomp_check_env(struct backend *backend) {
  const char *nomp_backend = get_if_env("NOMP_BACKEND");
  if (nomp_backend != NULL) {
    backend->backend =
        (char *)realloc(backend->backend, strnlen(nomp_backend, NOMP_BUFSIZ));
    strncpy(backend->backend, nomp_backend, strnlen(nomp_backend, NOMP_BUFSIZ));
    free((char *)nomp_backend);
  }
  int platform_id = strtoui(getenv("NOMP_PLATFORM_ID"), NOMP_BUFSIZ);
  if (platform_id >= 0)
    backend->platform_id = platform_id;
  int device_id = strtoui(getenv("NOMP_DEVICE_ID"), NOMP_BUFSIZ);
  if (device_id >= 0)
    backend->device_id = device_id;
  const char *install_dir = get_if_env("NOMP_INSTALL_DIR");
  if (install_dir != NULL) {
    envs.install_dir = (char *)malloc(strnlen(install_dir, NOMP_BUFSIZ));
    strncpy(envs.install_dir, install_dir,strnlen(install_dir, NOMP_BUFSIZ));
    free((char *)install_dir);
  }
  envs.verbose = strtoui(getenv("NOMP_VERBOSE_LEVEL"), NOMP_BUFSIZ);
}

static const char *py_dir = "python";

//=============================================================================
// nomp_init
//
int nomp_init(const char *backend, int platform, int device) {
  if (initialized)
    return NOMP_INITIALIZED_ERROR;

  nomp.backend = (char *)malloc(strnlen(backend, BUFSIZ));
  strncpy(nomp.backend, backend, strnlen(backend, BUFSIZ));
  nomp.platform_id = platform;
  nomp.device_id = device;
  nomp_check_env(&nomp);
  char name[NOMP_BUFSIZ];
  size_t n = strnlen(nomp.backend, NOMP_BUFSIZ);
  for (int i = 0; i < n; i++)
    name[i] = tolower(nomp.backend[i]);
  name[n] = '\0';

  int err = NOMP_INVALID_BACKEND;
  // FIXME: This is ugly -- should be fixed
#if defined(OPENCL_ENABLED)
  if (strncmp(name, "opencl", NOMP_BUFSIZ) == 0)
    err = opencl_init(&nomp, nomp.platform_id, nomp.device_id);
#endif
#if defined(CUDA_ENABLED)
  if (strncmp(name, "cuda", NOMP_BUFSIZ) == 0)
    err = cuda_init(&nomp, nomp.platform_id, nomp.device_id);
#endif
  if (err)
    return err;
  strncpy(nomp.name, name, NOMP_BUFSIZ);

  err = NOMP_PY_INITIALIZE_ERROR;
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
    if (envs.install_dir) {
      char *abs_dir = strcatn(3, envs.install_dir, "/", py_dir);
      py_append_to_sys_path(abs_dir);
      FREE(abs_dir);
      err = 0;
    }
  } else {
    // Python is already initialized.
    err = 0;
  }
  if (err)
    return err;

  initialized = 1;
  return 0;
}

//=============================================================================
// nomp_update
//
static struct mem **mems = NULL;
static int mems_n = 0;
static int mems_max = 0;

/// Returns the pointer to the allocated memory corresponding to 'p'.
/// If no buffer has been allocated for 'p' returns *mems_n*.
struct mem *mem_if_mapped(void *p) {
  // FIXME: This is O(N) in number of allocations.
  // Needs to go. Must store a hashmap.
  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i] && mems[i]->hptr == p)
      return mems[i];
  }
  return NULL;
}

static unsigned mem_if_exist(void *p, size_t idx0, size_t idx1) {
  // FIXME: This is O(N) in number of allocations.
  // Needs to go. Must store a hashmap.
  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i] && mems[i]->hptr == p && mems[i]->idx0 == idx0 &&
        mems[i]->idx1 == idx1)
      return i;
  }
  return mems_n;
}

int nomp_update(void *ptr, size_t idx0, size_t idx1, size_t usize, int op) {
  unsigned idx = mem_if_exist(ptr, idx0, idx1);
  if (idx == mems_n) {
    // A new entry can't be created with NOMP_FREE or NOMP_FROM
    if (op == NOMP_FROM || op == NOMP_FREE)
      return NOMP_INVALID_MAP_PTR;
    // NOMP_ALLOC is implied since this is a new entry
    op |= NOMP_ALLOC;
    if (mems_n == mems_max) {
      mems_max += mems_max / 2 + 1;
      mems = trealloc(struct mem *, mems, mems_max);
    }
    struct mem *m = mems[mems_n] = tcalloc(struct mem, 1);
    m->idx0 = idx0, m->idx1 = idx1, m->usize = usize;
    m->hptr = ptr, m->bptr = NULL;
  }

  int err = nomp.update(&nomp, mems[idx], op);

  // Device memory got free'd
  if (mems[idx]->bptr == NULL) {
    free(mems[idx]);
    mems[idx] = NULL;
  } else if (idx == mems_n) {
    mems_n++;
  }

  return err;
}

//=============================================================================
// nomp_jit
//
static struct prog **progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

static int parse_clauses(char **usr_file, char **usr_func,
                         const char **clauses) {
  // Currently, we only support `transform` and `jit`.
  unsigned i = 0;
  char *clause = NULL;
  while (clauses[i]) {
    strnlower(&clause, clauses[i], NOMP_BUFSIZ);
    if (strncmp(clause, "transform", NOMP_BUFSIZ) == 0) {
      char *val = strndup(clauses[i + 1], NOMP_BUFSIZ);
      char *tok = strtok(val, ":");
      if (tok) {
        *usr_file = strndup(tok, PATH_MAX), tok = strtok(NULL, ":");
        if (tok)
          *usr_func = strndup(tok, NOMP_BUFSIZ);
      }
      FREE(val);
    } else if (strncmp(clause, "jit", NOMP_BUFSIZ) == 0) {
    } else {
      FREE(clause);
      return NOMP_INVALID_CLAUSE;
    }
    i = i + 2;
  }
  FREE(clause);
  return 0;
}

int nomp_jit(int *id, const char *c_src, const char **annotations,
             const char **clauses) {
  if (*id == -1) {
    if (progs_n == progs_max) {
      progs_max += progs_max / 2 + 1;
      progs = trealloc(struct prog *, progs, progs_max);
    }

    // Create loopy kernel from C source
    PyObject *py_knl = NULL;
    int err = py_c_to_loopy(&py_knl, c_src, nomp.name);
    return_on_err(err);

    // Call the User callback function
    char *usr_file = NULL, *usr_func = NULL;
    err = parse_clauses(&usr_file, &usr_func, clauses);
    return_on_err(err);
    err = py_user_callback(&py_knl, usr_file, usr_func);
    FREE(usr_file);
    FREE(usr_func);
    return_on_err(err);

    // Get OpenCL, CUDA, etc. source and name from the loopy kernel
    char *name, *src;
    err = py_get_knl_name_and_src(&name, &src, py_knl);
    return_on_err(err);

    // Build the kernel
    struct prog *prg = progs[progs_n] = tcalloc(struct prog, 1);
    err = nomp.knl_build(&nomp, prg, src, name);
    FREE(src);
    FREE(name);
    return_on_err(err);

    // Get grid size of the loopy kernel as pymbolic expressions after
    // transformations. These grid sizes will be evaluated when the kernel is
    // run.
    prg->py_dict = PyDict_New();
    py_get_grid_size(prg, py_knl);
    Py_XDECREF(py_knl);

    if (err)
      return NOMP_KNL_BUILD_ERROR;
    *id = progs_n++;
  }

  return 0;
}

//=============================================================================
// nomp_run
//
int nomp_run(int id, int nargs, ...) {
  if (id >= 0) {
    struct prog *prg = progs[id];
    prg->nargs = nargs;

    va_list args;
    va_start(args, nargs);
    for (int i = 0; i < nargs; i++) {
      const char *var = va_arg(args, const char *);
      int type = va_arg(args, int);
      size_t size = va_arg(args, size_t);
      void *val = va_arg(args, void *);
      if (type == NOMP_INTEGER) {
        PyObject *py_key = PyUnicode_FromStringAndSize(var, strlen(var));
        PyObject *py_val = PyLong_FromLong(*((int *)val));
        PyDict_SetItem(prg->py_dict, py_key, py_val);
        Py_XDECREF(py_key), Py_XDECREF(py_val);
      }
    }
    va_end(args);
    py_eval_grid_size(prg, prg->py_dict);

    va_start(args, nargs);
    int err = nomp.knl_run(&nomp, prg, args);
    va_end(args);
    if (err)
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
  case NOMP_INVALID_MAP_PARAMS:
    strncpy(buf, "Invalid NOMP map parameters", buf_size);
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
    if (mems[i]) {
      nomp.update(&nomp, mems[i], NOMP_FREE);
      free(mems[i]);
      mems[i] = NULL;
    }
  }
  FREE(mems);
  mems = NULL, mems_n = mems_max = 0;

  for (unsigned i = 0; i < progs_n; i++) {
    if (progs[i]) {
      nomp.knl_free(progs[i]);
      free(progs[i]);
      progs[i] = NULL;
    }
  }
  FREE(progs);
  progs = NULL, progs_n = progs_max = 0;

  initialized = nomp.finalize(&nomp);
  free(nomp.backend), free(envs.install_dir);
  if (initialized)
    return NOMP_FINALIZE_ERROR;

  return 0;
}

//=============================================================================
// Helper functions: nomp_err & nomp_assert
//
void nomp_chk_(int err, const char *file, unsigned line) {
  if (err) {
    char buf[2 * NOMP_BUFSIZ];
    nomp_err(buf, err, 2 * NOMP_BUFSIZ);
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
