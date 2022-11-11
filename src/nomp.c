#include "nomp-impl.h"

int check_null_input_(void *p, const char *func, unsigned line,
                      const char *file) {
  if (!p) {
    return set_log(
        NOMP_NULL_INPUT, NOMP_ERROR,
        "Input pointer passed to function \"%s\" at line %d in fle %s is NULL.",
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

  tmp = get_if_env("NOMP_INSTALL_DIR");
  if (tmp != NULL) {
    size_t size = pathlen(tmp);
    backend->install_dir = tcalloc(char, size + 1);
    strncpy(backend->install_dir, tmp, size + 1), tfree(tmp);
  } else {
    // TODO: Default to ${HOME}/.nomp. If the doesn't exist, error out.
  }

  backend->verbose = strntoui(getenv("NOMP_VERBOSE_LEVEL"), NOMP_BUFSIZ);

  tmp = get_if_env("NOMP_ANNOTATE_SCRIPT");
  if (tmp) {
    size_t size = pathlen(tmp);
    backend->annts_script = tcalloc(char, size + 1);
    strncpy(backend->annts_script, tmp, size + 1), tfree(tmp);
  } else {
    // TODO: Default to current dir.
  }

  return 0;
}

static struct backend nomp;
static int initialized = 0;
static const char *py_dir = "python";

int nomp_init(const char *backend, int platform, int device) {
  if (initialized)
    return set_log(NOMP_INITIALIZED_ERROR, NOMP_ERROR,
                   ERR_STR_NOMP_IS_ALREADY_INITIALIZED, nomp.name);

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
    err = set_log(NOMP_INVALID_BACKEND, NOMP_ERROR,
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
    if (nomp.install_dir) {
      char *abs_dir = strcatn(3, nomp.install_dir, "/", py_dir);
      py_append_to_sys_path(abs_dir);
      err = tfree(abs_dir);
    } else {
      return set_log(NOMP_INSTALL_DIR_NOT_FOUND, NOMP_ERROR,
                     ERR_STR_NOMP_INSTALL_DIR_NOT_SET);
    }
  } else {
    // Python is already initialized.
    err = 0;
  }
  if (err)
    return set_log(NOMP_PY_INITIALIZE_ERROR, NOMP_ERROR,
                   ERR_STR_PY_INITIALIZE_ERROR);

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
      return set_log(NOMP_INVALID_MAP_PTR, NOMP_ERROR, ERR_STR_INVALID_MAP_OP,
                     op);
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

static int parse_clauses(char **usr_file, char **usr_func,
                         const char **clauses) {
  // Currently, we only support `transform` and `jit`.
  unsigned i = 0;
  char *clause = NULL;
  while (clauses[i]) {
    strnlower(&clause, clauses[i], NOMP_BUFSIZ);
    if (strncmp(clause, "transform", NOMP_BUFSIZ) == 0) {
      if (clauses[i + 1]) {
        size_t size = pathlen(clauses[i + 1]);
        *usr_file = strndup(clauses[i + 1], size);
      } else {
        tfree(clause);
        return set_log(NOMP_FILE_NAME_NOT_PROVIDED, NOMP_ERROR,
                       ERR_STR_FILE_NAME_NOT_PROVIDED);
      }
      if (clauses[i + 2]) {
        *usr_func = strndup(clauses[i + 2], NOMP_BUFSIZ);
      } else {
        tfree(clause);
        return set_log(NOMP_USER_CALLBACK_NOT_PROVIDED, NOMP_ERROR,
                       ERR_STR_USER_CALLBACK_NOT_PROVIDED);
      }
      i = i + 3;
    } else if (strncmp(clause, "jit", NOMP_BUFSIZ) == 0) {
    } else {
      tfree(clause);
      return set_log(NOMP_INVALID_CLAUSE, NOMP_ERROR,
                     ERR_STR_NOMP_INVALID_CLAUSE, clauses[i]);
    }
  }
  tfree(clause);
  return 0;
}

static int annts_to_dict(PyObject *dict, const char **annts) {
  unsigned i = 0;
  while (annts[i]) {
    PyObject *key =
        PyUnicode_FromStringAndSize(annts[i], strnlen(annts[i], NOMP_BUFSIZ));
    PyObject *val = PyUnicode_FromStringAndSize(
        annts[i + 1], strnlen(annts[i + 1], NOMP_BUFSIZ));
    PyDict_SetItem(dict, key, val);
    Py_XDECREF(key), Py_XDECREF(val);
    i += 2;
  }

  return 0;
}

int nomp_jit(int *id, const char *c_src, const char **annts,
             const char **clauses) {
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

    // Handle annotations
    PyObject *annts_dict = PyDict_New();
    err = annts_to_dict(annts_dict, annts);
    return_on_err(err);
    // err = py_user_annotate(&knl, dict, nomp->annts_script, "annotate");
    // return_on_err(err);
    Py_XDECREF(annts_dict);

    // Parse the clauses
    char *usr_file = NULL, *usr_func = NULL;
    err = parse_clauses(&usr_file, &usr_func, clauses);
    return_on_err(err);

    // Call the User callback function
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
    py_get_grid_size(prg, knl);
    Py_XDECREF(knl);

    if (err)
      return set_log(NOMP_KNL_BUILD_ERROR, NOMP_ERROR, ERR_STR_KNL_BUILD_ERROR);
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
    py_eval_grid_size(prg, prg->py_dict);

    va_start(args, nargs);
    int err = nomp.knl_run(&nomp, prg, args);
    va_end(args);
    if (err)
      return set_log(NOMP_KNL_RUN_ERROR, NOMP_ERROR, ERR_STR_KERNEL_RUN_FAILED,
                     id);
    return 0;
  }
  return set_log(NOMP_INVALID_KNL, NOMP_ERROR, ERR_STR_INVALID_KERNEL, id);
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
    return set_log(NOMP_NOT_INITIALIZED_ERROR, NOMP_ERROR,
                   ERR_STR_NOMP_IS_NOT_INITIALIZED);

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

  tfree(nomp.backend), tfree(nomp.install_dir), tfree(nomp.annts_script);

  initialized = nomp.finalize(&nomp);
  if (initialized)
    return set_log(NOMP_FINALIZE_ERROR, NOMP_ERROR,
                   ERR_STR_FAILED_TO_FINALIZE_NOMP);

  return 0;
}
