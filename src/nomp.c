#include "nomp-impl.h"

int check_null_input_(void *p, const char *func, unsigned line,
                      const char *file) {
  if (!p) {
    return set_log(NOMP_RUNTIME_NULL_INPUT_ENCOUNTERED, NOMP_ERROR,
                   "Input pointer passed to function \"%s\" at line %d in file "
                   "%s is NULL.",
                   func, line, file);
  }
  return 0;
}

static char *get_if_env(const char *name) {
  const char *tmp = getenv(name);
  if (tmp != NULL) {
    char *tmp_var = tcalloc(char, NOMP_BUFSIZ);
    if (tmp_var != NULL) {
      strncpy(tmp_var, tmp, NOMP_BUFSIZ);
      return tmp_var;
    }
  }
  return NULL;
}

static int check_env(struct backend *backend) {
  char *tmp = get_if_env("NOMP_BACKEND");
  if (tmp != NULL) {
    backend->backend = trealloc(backend->backend, char, MAX_BACKEND_NAME_SIZE);
    strncpy(backend->backend, tmp, MAX_BACKEND_NAME_SIZE), tfree(tmp);
  }

  int platform_id = strntoui(getenv("NOMP_PLATFORM_ID"), NOMP_BUFSIZ);
  if (platform_id >= 0)
    backend->platform_id = platform_id;

  int device_id = strntoui(getenv("NOMP_DEVICE_ID"), NOMP_BUFSIZ);
  if (device_id >= 0)
    backend->device_id = device_id;

  // FIXME: We should get rid of these defaults after implementing MPI_Init
  // like arguments parsing for nomp_init().
  tmp = get_if_env("NOMP_INSTALL_DIR");
  if (tmp != NULL) {
    size_t size = pathlen(tmp);
    backend->install_dir = tcalloc(char, size + 1);
    strncpy(backend->install_dir, tmp, size), tfree(tmp);
  } else {
    // Default to ${HOME}/.nomp. Also, there is a way to find the directory
    // where the libnomp.so is located within linomp.so itself. Maybe we can do
    // so in a portable manner.
    const char *home = getenv("HOME");
    if (home)
      backend->install_dir = strcatn(2, home, "/.nomp");
    else
      return set_log(
          NOMP_USER_INPUT_NOT_PROVIDED, NOMP_ERROR,
          "Unable to initialize libnomp install directory. Neither "
          "NOMP_INSTALL_DIR nor HOME environment variables are defined.");
  }

  backend->verbose = strntoui(getenv("NOMP_VERBOSE_LEVEL"), NOMP_BUFSIZ);

  tmp = get_if_env("NOMP_ANNOTATE_SCRIPT");
  if (tmp) {
    size_t size = pathlen(tmp);
    backend->annts_script = tcalloc(char, size + 1);
    strncpy(backend->annts_script, tmp, size), tfree(tmp);
  }

  tmp = get_if_env("NOMP_ANNOTATE_FUNCTION");
  if (tmp) {
    size_t size = strnlen(tmp, NOMP_BUFSIZ);
    backend->annts_func = tcalloc(char, size + 1);
    strncpy(backend->annts_func, tmp, size), tfree(tmp);
  }

  return 0;
}

static struct backend nomp;
static int initialized = 0;
static const char *py_dir = "python";

int nomp_init(const char *backend, int platform, int device) {
  if (initialized)
    return set_log(
        NOMP_RUNTIME_ALREADY_INITIALIZED, NOMP_ERROR,
        "libnomp is already initialized to use %s. Call nomp_finalize() before "
        "calling nomp_init() again.",
        nomp.name);

  nomp.backend = tcalloc(char, MAX_BACKEND_NAME_SIZE);
  strncpy(nomp.backend, backend, MAX_BACKEND_NAME_SIZE);
  nomp.platform_id = platform, nomp.device_id = device;

  int err = check_env(&nomp);
  return_on_err(err);

  char name[MAX_BACKEND_NAME_SIZE];
  size_t n = strnlen(nomp.backend, MAX_BACKEND_NAME_SIZE);
  for (int i = 0; i < n; i++)
    name[i] = tolower(nomp.backend[i]);
  name[n] = '\0';

  if (strncmp(name, "opencl", MAX_BACKEND_NAME_SIZE) == 0) {
#if defined(OPENCL_ENABLED)
    err = opencl_init(&nomp, nomp.platform_id, nomp.device_id);
#endif
  } else if (strncmp(name, "cuda", MAX_BACKEND_NAME_SIZE) == 0) {
#if defined(CUDA_ENABLED)
    err = cuda_init(&nomp, nomp.platform_id, nomp.device_id);
#endif
  } else {
    err = set_log(NOMP_USER_INPUT_NOT_VALID, NOMP_ERROR,
                  "Failed to initialized libnomp. Invalid backend: %s", name);
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
    // nomp.install_dir should be set and we use it here.
    char *abs_dir = strcatn(3, nomp.install_dir, "/", py_dir);
    py_append_to_sys_path(abs_dir);
    err = tfree(abs_dir);
  } else {
    // Python is already initialized.
    err = 0;
  }
  if (err)
    return set_log(NOMP_PY_INITIALIZE_ERROR, NOMP_ERROR,
                   "Unable to initialize python during initializing libnomp.");

  initialized = 1;
  return 0;
}

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
      return set_log(NOMP_USER_MAP_OP_NOT_VALID, NOMP_ERROR,
                     "NOMP_FREE or NOMP_FROM can only be called on a pointer "
                     "which is already on the device.");
    op |= NOMP_ALLOC;
    if (mems_n == mems_max) {
      mems_max += mems_max / 2 + 1;
      mems = trealloc(mems, struct mem *, mems_max);
    }
    struct mem *m = mems[mems_n] = tcalloc(struct mem, 1);
    m->idx0 = idx0, m->idx1 = idx1, m->usize = usize;
    m->hptr = ptr, m->bptr = NULL;
  }

  int err = nomp.update(&nomp, mems[idx], op);

  // Device memory got free'd
  if (mems[idx]->bptr == NULL)
    tfree(mems[idx]), mems[idx] = NULL;
  else if (idx == mems_n)
    mems_n++;

  return err;
}

static struct prog **progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

static int parse_clauses(char **usr_file, char **usr_func, PyObject **dict_,
                         const char **clauses) {
  // Currently, we only support `transform` and `annotate` and `jit`.
  PyObject *dict = *dict_ = PyDict_New();
  unsigned i = 0;
  while (clauses[i]) {
    if (strncmp(clauses[i], "transform", NOMP_BUFSIZ) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL)
        return set_log(
            NOMP_USER_INPUT_NOT_PROVIDED, NOMP_ERROR,
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");
      *usr_file = strndup(clauses[i + 1], pathlen(clauses[i + 1]));
      *usr_func = strndup(clauses[i + 2], NOMP_BUFSIZ);
      i = i + 3;
    } else if (strncmp(clauses[i], "annotate", NOMP_BUFSIZ) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL)
        return set_log(NOMP_USER_INPUT_NOT_PROVIDED, NOMP_ERROR,
                       "\"annotate\" clause should be followed by a key value "
                       "pair. At least one of them is not provided.");
      const char *key = clauses[i + 1], *val = clauses[i + 2];
      PyObject *pkey =
          PyUnicode_FromStringAndSize(key, strnlen(key, NOMP_BUFSIZ));
      PyObject *pval =
          PyUnicode_FromStringAndSize(val, strnlen(val, NOMP_BUFSIZ));
      PyDict_SetItem(dict, pkey, pval);
      Py_XDECREF(pkey), Py_XDECREF(pval);
      i = i + 3;
    } else {
      return set_log(
          NOMP_USER_INPUT_NOT_VALID, NOMP_ERROR,
          "Clause \"%s\" passed into nomp_jit is not a valid caluse.",
          clauses[i]);
    }
  }

  return 0;
}

int nomp_jit(int *id, const char *c_src, const char **clauses) {
  int err;
  if (*id == -1) {
    if (progs_n == progs_max) {
      progs_max += progs_max / 2 + 1;
      progs = trealloc(progs, struct prog *, progs_max);
    }

    // Create loopy kernel from C source
    PyObject *knl = NULL;
    int err = py_c_to_loopy(&knl, c_src, nomp.name);
    return_on_err(err);

    // Parse the clauses
    char *usr_file = NULL, *usr_func = NULL;
    PyObject *annts;
    err = parse_clauses(&usr_file, &usr_func, &annts, clauses);
    return_on_err(err);

    // Handle annotate clauses if the exist
    err = py_user_annotate(&knl, annts, nomp.annts_script, nomp.annts_func);
    return_on_err(err);

    // Handle transform clauase
    err = py_user_transform(&knl, usr_file, usr_func);
    tfree(usr_file), tfree(usr_func);
    return_on_err(err);

    // Get OpenCL, CUDA, etc. source and name from the loopy kernel
    char *name, *src;
    err = py_get_knl_name_and_src(&name, &src, knl);
    return_on_err(err);

    // Build the kernel
    struct prog *prg = progs[progs_n] = tcalloc(struct prog, 1);
    err = nomp.knl_build(&nomp, prg, src, name);
    tfree(src), tfree(name);
    return_on_err(err);

    // Get grid size of the loopy kernel as pymbolic expressions after
    // transformations. These grid sizes will be evaluated when the kernel is
    // run.
    prg->py_dict = PyDict_New();
    err = py_get_grid_size(prg, knl);
    Py_XDECREF(knl);
    return_on_err(err);

    *id = progs_n++;
  }

  return 0;
}

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
    int err = py_eval_grid_size(prg, prg->py_dict);
    return_on_err(err);

    va_start(args, nargs);
    err = nomp.knl_run(&nomp, prg, args);
    va_end(args);
    return_on_err(err);

    return 0;
  }
  return set_log(NOMP_USER_INPUT_NOT_VALID, NOMP_ERROR,
                 "Kernel id %d passed to nomp_run is not valid.", id);
}

void nomp_assert_(int cond, const char *file, unsigned line) {
  if (!cond) {
    printf("nomp_assert failure at %s:%d\n", file, line);
    exit(1);
  }
}

void nomp_chk_(int err_id, const char *file, unsigned line) {
  if (err_id == 0)
    return;
  if (nomp_get_log_type(err_id) == NOMP_ERROR) {
    char *err_str;
    int err = nomp_get_log_str(&err_str, err_id);
    printf("%s:%d %s\n", file, line, err_str);
    tfree(err_str);
    exit(1);
  }
}

int nomp_finalize(void) {
  if (!initialized)
    return set_log(NOMP_RUNTIME_NOT_INITIALIZED, NOMP_ERROR,
                   "libnomp is not initialized.");

  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i]) {
      // FIXME: Check error returned form `nomp.update`
      nomp.update(&nomp, mems[i], NOMP_FREE);
      tfree(mems[i]), mems[i] = NULL;
    }
  }
  tfree(mems), mems = NULL, mems_n = mems_max = 0;

  for (unsigned i = 0; i < progs_n; i++) {
    if (progs[i]) {
      // FIXME: Check error returned form `nomp.knl_free`
      nomp.knl_free(progs[i]);
      tfree(progs[i]), progs[i] = NULL;
    }
  }
  tfree(progs), progs = NULL, progs_n = progs_max = 0;

  tfree(nomp.backend), tfree(nomp.install_dir);
  tfree(nomp.annts_script), tfree(nomp.annts_func);

  initialized = nomp.finalize(&nomp);
  if (initialized)
    return set_log(NOMP_RUNTIME_FAILED_TO_FINALIZE, NOMP_ERROR,
                   "Failed to initialize libnomp.");

  return 0;
}
