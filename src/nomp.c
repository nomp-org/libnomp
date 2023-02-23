#include "nomp-impl.h"

static char *copy_env(const char *name, size_t size) {
  const char *tmp = getenv(name);
  if (tmp != NULL) {
    char *copy = nomp_calloc(char, size);
    if (copy != NULL) {
      strncpy(copy, tmp, size);
      return copy;
    }
  }
  return NULL;
}

static int check_env(struct backend *backend) {
  char *tmp = getenv("NOMP_PLATFORM_ID");
  if (tmp)
    backend->platform_id = strntoui(tmp, MAX_BUFSIZ);

  tmp = getenv("NOMP_DEVICE_ID");
  if (tmp)
    backend->device_id = strntoui(tmp, MAX_BUFSIZ);

  tmp = getenv("NOMP_VERBOSE_LEVEL");
  if (tmp)
    backend->verbose = strntoui(tmp, MAX_BUFSIZ);

  tmp = copy_env("NOMP_BACKEND", MAX_BACKEND_NAME_SIZE);
  if (tmp) {
    backend->backend = trealloc(backend->backend, char, MAX_BACKEND_NAME_SIZE);
    strncpy(backend->backend, tmp, MAX_BACKEND_NAME_SIZE), nomp_free(tmp);
  }

  tmp = copy_env("NOMP_ANNOTATE_SCRIPT", MAX_BUFSIZ);
  if (tmp) {
    size_t size = strnlen(tmp, MAX_BUFSIZ) + 1;
    backend->annts_script = trealloc(backend->annts_script, char, size);
    strncpy(backend->annts_script, tmp, size), nomp_free(tmp);
  }

  tmp = copy_env("NOMP_ANNOTATE_FUNCTION", MAX_BUFSIZ);
  if (tmp) {
    size_t size = strnlen(tmp, MAX_BUFSIZ) + 1;
    backend->annts_func = trealloc(backend->annts_func, char, size);
    strncpy(backend->annts_func, tmp, size), nomp_free(tmp);
  }

  tmp = copy_env("NOMP_INSTALL_DIR", MAX_BUFSIZ);
  if (tmp) {
    size_t size;
    return_on_err(pathlen(&size, tmp));
    backend->install_dir = trealloc(backend->install_dir, char, size + 1);
    strncpy(backend->install_dir, tmp, size), nomp_free(tmp);
  }

  return 0;
}

static struct backend nomp;
static int initialized = 0;
static const char *py_dir = "python";

static int check_args(int argc, const char **argv, struct backend *backend) {
  backend->device_id = 0, backend->platform_id = 0, backend->verbose = 0;
  backend->backend = backend->install_dir = NULL;
  backend->annts_script = backend->annts_func = NULL;

  if (argc <= 1 || argv == NULL)
    return 0;

  unsigned i = 0;
  while (i < argc) {
    if (strncmp("-", argv[i], 1)) {
      i += 1;
      continue;
    }
    if (i + 1 == argc) {
      return set_log(
          NOMP_USER_ARG_IS_INVALID, NOMP_ERROR,
          strcatn(2, MAX_BUFSIZ, "Missing argument value: ", argv[i]));
    }

    if (!strncmp("-b", argv[i], MAX_BACKEND_NAME_SIZE) ||
        !strncmp("--backend", argv[i], MAX_BACKEND_NAME_SIZE)) {
      if (argv[i + 1])
        backend->backend =
            strndup((const char *)argv[i + 1], MAX_BACKEND_NAME_SIZE);
      i += 2;
    } else if (!strncmp("-p", argv[i], MAX_BUFSIZ) ||
               !strncmp("--platform", argv[i], MAX_BUFSIZ)) {
      backend->platform_id = strntoui(argv[i + 1], MAX_BUFSIZ);
      i += 2;
    } else if (!strncmp("-d", argv[i], MAX_BUFSIZ) ||
               !strncmp("--device", argv[i], MAX_BUFSIZ)) {
      backend->device_id = strntoui(argv[i + 1], MAX_BUFSIZ);
      i += 2;
    } else if (!strncmp("-v", argv[i], MAX_BUFSIZ) ||
               !strncmp("--verbose", argv[i], MAX_BUFSIZ)) {
      backend->verbose = strntoui(argv[i + 1], MAX_BUFSIZ);
      i += 2;
    } else if (!strncmp("-i", argv[i], MAX_BUFSIZ) ||
               !strncmp("--install-dir", argv[i], MAX_BUFSIZ)) {
      char *install_dir = (char *)argv[i + 1];
      size_t size;
      return_on_err(pathlen(&size, install_dir));
      backend->install_dir = strndup(install_dir, size + 1);
      i += 2;
    } else if (!strncmp("-as", argv[i], MAX_BUFSIZ) ||
               !strncmp("--annts-script", argv[i], MAX_BUFSIZ)) {
      if (argv[i + 1])
        backend->annts_script = strndup((const char *)argv[i + 1], MAX_BUFSIZ);
      i += 2;
    } else if (!strncmp("-af", argv[i], MAX_BUFSIZ) ||
               !strncmp("--annts-func", argv[i], MAX_BUFSIZ)) {
      if (argv[i + 1])
        backend->annts_func = strndup((const char *)argv[i + 1], MAX_BUFSIZ);
      i += 2;
    } else {
      return set_log(NOMP_USER_ARG_IS_INVALID, NOMP_ERROR,
                     strcatn(2, MAX_BUFSIZ, "Invalid argument : ", argv[i]));
    }
  }

  return 0;
}

int nomp_init(int argc, const char **argv) {
  if (initialized) {
    return set_log(NOMP_INITIALIZE_FAILURE, NOMP_ERROR,
                   "libnomp is already initialized.");
  }

  return_on_err(check_args(argc, argv, &nomp));
  return_on_err(check_env(&nomp));

  char name[MAX_BACKEND_NAME_SIZE + 1];
  size_t n = strnlen(nomp.backend, MAX_BACKEND_NAME_SIZE);
  for (int i = 0; i < n; i++)
    name[i] = tolower(nomp.backend[i]);
  name[n] = '\0';

  int err = 1;
  if (strncmp(name, "opencl", MAX_BACKEND_NAME_SIZE) == 0) {
#if defined(OPENCL_ENABLED)
    err = opencl_init(&nomp, nomp.platform_id, nomp.device_id);
#endif
  } else if (strncmp(name, "cuda", MAX_BACKEND_NAME_SIZE) == 0) {
#if defined(CUDA_ENABLED)
    err = cuda_init(&nomp, nomp.platform_id, nomp.device_id);
#endif
  } else {
    err = set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                  "Failed to initialized libnomp. Invalid backend: %s", name);
  }
  return_on_err(err);

  strncpy(nomp.name, name, MAX_BACKEND_NAME_SIZE);

  if (!Py_IsInitialized()) {
    // May be we need the isolated configuration listed here:
    // https://docs.python.org/3/c-api/init_config.html#init-config
    // But for now, we do the simplest thing possible.
    Py_Initialize();

    // Append current working directroy to sys.path.
    return_on_err(py_append_to_sys_path("."));

    // Append nomp python directory to sys.path.
    // nomp.install_dir should be set and we use it here.
    size_t len;
    return_on_err(pathlen(&len, nomp.install_dir));
    len = maxn(2, len, strnlen(py_dir, MAX_BUFSIZ));
    char *abs_dir = strcatn(3, len, nomp.install_dir, "/", py_dir);
    return_on_err(py_append_to_sys_path(abs_dir));

    nomp_free(abs_dir);
  }

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
    if (op == NOMP_FROM || op == NOMP_FREE) {
      return set_log(NOMP_USER_MAP_OP_IS_INVALID, NOMP_ERROR,
                     "NOMP_FREE or NOMP_FROM can only be called on a pointer "
                     "which is already on the device.");
    }
    op |= NOMP_ALLOC;
    if (mems_n == mems_max) {
      mems_max += mems_max / 2 + 1;
      mems = nomp_realloc(mems, struct mem *, mems_max);
    }
    struct mem *m = mems[mems_n] = nomp_calloc(struct mem, 1);
    m->idx0 = idx0, m->idx1 = idx1, m->usize = usize;
    m->hptr = ptr, m->bptr = NULL;
  }

  return_on_err(nomp.update(&nomp, mems[idx], op));

  // Device memory object was free'd
  if (mems[idx]->bptr == NULL)
    nomp_free(mems[idx]), mems[idx] = NULL;
  // Or new memory object got created
  else if (idx == mems_n)
    mems_n++;

  return 0;
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
    if (strncmp(clauses[i], "transform", MAX_BUFSIZ) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return set_log(
            NOMP_USER_INPUT_NOT_PROVIDED, NOMP_ERROR,
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");
      }
      char *file = strcatn(2, PATH_MAX, (const char *)clauses[i + 1], ".py");
      return_on_err(pathlen(NULL, file));
      tfree(file);
      *usr_file = strndup(clauses[i + 1], PATH_MAX);
      *usr_func = strndup(clauses[i + 2], MAX_BUFSIZ);
      i += 3;
    } else if (strncmp(clauses[i], "annotate", MAX_BUFSIZ) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return set_log(NOMP_USER_INPUT_NOT_PROVIDED, NOMP_ERROR,
                       "\"annotate\" clause should be followed by a key value "
                       "pair. At least one of them is not provided.");
      }
      const char *key = clauses[i + 1], *val = clauses[i + 2];
      PyObject *pkey =
          PyUnicode_FromStringAndSize(key, strnlen(key, MAX_BUFSIZ));
      PyObject *pval =
          PyUnicode_FromStringAndSize(val, strnlen(val, MAX_BUFSIZ));
      PyDict_SetItem(dict, pkey, pval);
      Py_XDECREF(pkey), Py_XDECREF(pval);
      i += 3;
    } else {
      return set_log(
          NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
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
      progs = nomp_realloc(progs, struct prog *, progs_max);
    }

    // Create loopy kernel from C source
    PyObject *knl = NULL;
    return_on_err(py_c_to_loopy(&knl, c_src, nomp.name));

    // Parse the clauses
    char *usr_file = NULL, *usr_func = NULL;
    PyObject *annts;
    return_on_err(parse_clauses(&usr_file, &usr_func, &annts, clauses));

    // Handle annotate clauses if the exist
    return_on_err(
        py_user_annotate(&knl, annts, nomp.annts_script, nomp.annts_func));
    Py_XDECREF(annts);

    // Handle transform clauase
    return_on_err(py_user_transform(&knl, usr_file, usr_func));
    nomp_free(usr_file), nomp_free(usr_func);

    // Get OpenCL, CUDA, etc. source and name from the loopy kernel
    char *name, *src;
    return_on_err(py_get_knl_name_and_src(&name, &src, knl));

    // Build the kernel
    struct prog *prg = progs[progs_n] = nomp_calloc(struct prog, 1);
    return_on_err(nomp.knl_build(&nomp, prg, src, name));
    nomp_free(src), nomp_free(name);

    // Get grid size of the loopy kernel as pymbolic expressions after
    // transformations. These grid sizes will be evaluated when the kernel is
    // run.
    prg->py_dict = PyDict_New();
    return_on_err(py_get_grid_size(prg, knl));
    Py_XDECREF(knl);

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
      if (type == NOMP_INT) {
        PyObject *py_key = PyUnicode_FromStringAndSize(var, strlen(var));
        PyObject *py_val = PyLong_FromLong(*((int *)val));
        PyDict_SetItem(prg->py_dict, py_key, py_val);
        Py_XDECREF(py_key), Py_XDECREF(py_val);
      }
    }
    va_end(args);
    return_on_err(py_eval_grid_size(prg, prg->py_dict));

    va_start(args, nargs);
    return_on_err(nomp.knl_run(&nomp, prg, args));
    va_end(args);

    return 0;
  }

  return set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                 "Kernel id %d passed to nomp_run is not valid.", id);
}

int nomp_sync() { return nomp.sync(&nomp); }

void nomp_chk(int retval) {
  if (nomp_get_log_type(retval) == NOMP_ERROR) {
    fprintf(stderr, "%s\n", nomp_get_log_str(retval));
    exit(1);
  }
}

int nomp_finalize(void) {
  if (!initialized) {
    return set_log(NOMP_FINALIZE_FAILURE, NOMP_ERROR,
                   "libnomp is not initialized.");
  }

  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i]) {
      return_on_err(nomp.update(&nomp, mems[i], NOMP_FREE));
      nomp_free(mems[i]), mems[i] = NULL;
    }
  }
  nomp_free(mems), mems = NULL, mems_n = mems_max = 0;

  for (unsigned i = 0; i < progs_n; i++) {
    if (progs[i]) {
      return_on_err(nomp.knl_free(progs[i]));
      nomp_free(progs[i]), progs[i] = NULL;
    }
  }
  nomp_free(progs), progs = NULL, progs_n = progs_max = 0;

  nomp_free(nomp.backend), nomp_free(nomp.install_dir);
  nomp_free(nomp.annts_script), nomp_free(nomp.annts_func);

  initialized = nomp.finalize(&nomp);
  if (initialized) {
    return set_log(NOMP_FINALIZE_FAILURE, NOMP_ERROR,
                   "Failed to initialize libnomp.");
  }

  return 0;
}
