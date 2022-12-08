#include "nomp-impl.h"

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
    size_t size = pathlen(tmp) + 1;
    backend->install_dir = tcalloc(char, size);
    strncpy(backend->install_dir, tmp, size), tfree(tmp);
  } else {
    // Default to ${HOME}/.nomp. Also, there is a way to find the directory
    // where the libnomp.so is located within linomp.so itself. Maybe we can do
    // so in a portable manner.
    const char *home = getenv("HOME");
    if (home)
      backend->install_dir =
          strcatn(2, MAX(2, pathlen(home), NOMP_BUFSIZ), home, "/.nomp");
    else
      return set_log(
          NOMP_USER_INPUT_NOT_PROVIDED, NOMP_ERROR,
          "Unable to initialize libnomp install directory. Neither "
          "NOMP_INSTALL_DIR nor HOME environment variables are defined.");
  }

  backend->verbose = strntoui(getenv("NOMP_VERBOSE_LEVEL"), NOMP_BUFSIZ);

  tmp = get_if_env("NOMP_ANNOTATE_SCRIPT");
  if (tmp) {
    size_t size = strnlen(tmp, NOMP_BUFSIZ) + 1;
    backend->script = tcalloc(char, size);
    strncpy(backend->script, tmp, size), tfree(tmp);
  }

  tmp = get_if_env("NOMP_ANNOTATE_FUNCTION");
  if (tmp) {
    size_t size = strnlen(tmp, NOMP_BUFSIZ) + 1;
    backend->annts_func = tcalloc(char, size);
    strncpy(backend->annts_func, tmp, size), tfree(tmp);
  }

  return 0;
}

static struct backend nomp;
static int initialized = 0;
static const char *py_dir = "python";

int check_args(int argc, const char **argv, struct backend *backend) {
  backend->backend = "opencl";
  backend->device_id = 0, backend->platform_id = 0, backend->verbose = 0;
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
          strcatn(2, NOMP_BUFSIZ, "Missing argument value: ", argv[i]));
    }

    if (!strncmp("-b", argv[i], MAX_BACKEND_NAME_SIZE) ||
        !strncmp("--backend", argv[i], MAX_BACKEND_NAME_SIZE)) {
      backend->backend = strndup(argv[i + 1], MAX_BACKEND_NAME_SIZE);
      i += 2;
    } else if (!strncmp("-p", argv[i], NOMP_BUFSIZ) ||
               !strncmp("--platform", argv[i], NOMP_BUFSIZ)) {
      int platform_id = strntoui(argv[i + 1], NOMP_BUFSIZ);
      if (platform_id >= 0)
        backend->platform_id = platform_id;
      i += 2;
    } else if (!strncmp("-d", argv[i], NOMP_BUFSIZ) ||
               !strncmp("--device", argv[i], NOMP_BUFSIZ)) {
      int device_id = strntoui(argv[i + 1], NOMP_BUFSIZ);
      if (device_id >= 0)
        backend->device_id = device_id;
      i += 2;
    } else if (!strncmp("-v", argv[i], NOMP_BUFSIZ) ||
               !strncmp("--verbose", argv[i], NOMP_BUFSIZ)) {
      int verbose = strntoui(argv[i + 1], NOMP_BUFSIZ);
      if (verbose >= 0)
        backend->verbose = strntoui(argv[i + 1], NOMP_BUFSIZ);
      i += 2;
    } else if (!strncmp("-i", argv[i], NOMP_BUFSIZ) ||
               !strncmp("--install-dir", argv[i], NOMP_BUFSIZ)) {
      char *install_dir = (char *)argv[i + 1];
      size_t size = pathlen(install_dir) + 1;
      if (install_dir != NULL)
        backend->install_dir = strndup(install_dir, size);
      i += 2;
    } else if (!strncmp("-as", argv[i], NOMP_BUFSIZ) ||
               !strncmp("--annts-script", argv[i], NOMP_BUFSIZ)) {
      char *annts_script = (char *)argv[i + 1];
      if (annts_script != NULL)
        backend->annts_script = strndup(annts_script, NOMP_BUFSIZ);
      i += 2;
    } else if (!strncmp("-af", argv[i], NOMP_BUFSIZ) ||
               !strncmp("--annts-func", argv[i], NOMP_BUFSIZ)) {
      char *annts_func = (char *)argv[i + 1];
      if (annts_func != NULL) {
        backend->annts_func = strndup(annts_func, NOMP_BUFSIZ);
      }
      i += 2;
    } else {
      return set_log(NOMP_USER_ARG_IS_INVALID, NOMP_ERROR,
                     strcatn(2, NOMP_BUFSIZ, "Invalid argument : ", argv[i]));
    }
  }

  return 0;
}

int nomp_init(int argc, const char **argv) {
  if (initialized) {
    return set_log(
        NOMP_RUNTIME_ALREADY_INITIALIZED, NOMP_ERROR,
        "libnomp is already initialized to use %s. Call nomp_finalize() before "
        "calling nomp_init() again.",
        nomp.backend);

  int err = check_args(argc, argv, &nomp);
  return_on_err(err);
  err = check_env(&nomp);
  return_on_err(err);

  char name[MAX_BACKEND_NAME_SIZE + 1];
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
    err = set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                  "Failed to initialized libnomp. Invalid backend: %s", name);
  }
  return_on_err(err);

  if (!Py_IsInitialized()) {
    // May be we need the isolated configuration listed here:
    // https://docs.python.org/3/c-api/init_config.html#init-config
    // But for now, we do the simplest thing possible.
    Py_Initialize();
    // Append current working dir
    py_append_to_sys_path(".");
    // nomp.install_dir should be set and we use it here.
    char *abs_dir = strcatn(
        3, MAX(2, pathlen(nomp.install_dir), strnlen(py_dir, NOMP_BUFSIZ)),
        nomp.install_dir, "/", py_dir);
    py_append_to_sys_path(abs_dir);
    err = tfree(abs_dir);
  } else {
    // Python is already initialized.
    err = 0;
  }
  if (err) {
    return set_log(NOMP_PY_INITIALIZE_ERROR, NOMP_ERROR,
                   "Unable to initialize python during initializing libnomp.");
  }

  // Allocate buffer for use as temporary arguments for kernels (e.g., like
  // reductions).
  struct mem *m = nomp.buffer = tcalloc(struct mem, 1);
  m->idx0 = 0, m->idx1 = 1024, m->usize = sizeof(char);
  m->hptr = tmalloc(char, 1024);
  m->bptr = NULL;
  err = nomp.update(&nomp, m, NOMP_ALLOC);
  return_on_err(err);

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
      return set_log(NOMP_USER_MAP_OP_IS_INVALID, NOMP_ERROR,
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

static void free_meta(struct meta *info) {
  tfree(info->file), tfree(info->func);
  Py_XDECREF(info->dict);
}

static int parse_clauses(struct meta *info, const char **clauses) {
  info->file = info->func = NULL;
  info->dict = PyDict_New();

  unsigned i = 0;
  while (clauses[i]) {
    if (strncmp(clauses[i], "transform", NOMP_BUFSIZ) == 0) {
      // Syntax: "transform", <script-name>, <function-name>
      if (!clauses[i + 1] || !clauses[i + 2]) {
        return set_log(
            NOMP_USER_INPUT_NOT_PROVIDED, NOMP_ERROR,
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");
      }
      info->file = strndup(clauses[i + 1], pathlen(clauses[i + 1]));
      info->func = strndup(clauses[i + 2], NOMP_BUFSIZ);
      i = i + 3;
    } else if (strncmp(clauses[i], "annotate", NOMP_BUFSIZ) == 0) {
      // Syntax: "annotate", <key>, <value>
      if (!clauses[i + 1] || !clauses[i + 2]) {
        return set_log(NOMP_USER_INPUT_NOT_PROVIDED, NOMP_ERROR,
                       "\"annotate\" clause should be followed by a key value "
                       "pair. At least one of them is not provided.");
      }
      const char *key = clauses[i + 1], *val = clauses[i + 2];
      PyObject *pkey =
          PyUnicode_FromStringAndSize(key, strnlen(key, NOMP_BUFSIZ));
      PyObject *pval =
          PyUnicode_FromStringAndSize(val, strnlen(val, NOMP_BUFSIZ));
      PyDict_SetItem(info->dict, pkey, pval);
      Py_XDECREF(pkey), Py_XDECREF(pval);
      i = i + 3;
    } else {
      return set_log(
          NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
          "Clause \"%s\" passed into nomp_jit is not a valid caluse.",
          clauses[i]);
    }
  }

  return 0;
}

int nomp_jit(int *id, const char *c_src, const char **clauses, int narg, ...) {
  if (*id == -1) {
    if (progs_n == progs_max) {
      progs_max += progs_max / 2 + 1;
      progs = trealloc(progs, struct prog *, progs_max);
    }
    struct prog *prg = progs[progs_n] = tcalloc(struct prog, 1);

    prg->reduction_indx = prg->reduction_op = -1;
    prg->narg = narg;
    prg->args = tcalloc(struct arg, narg);

    // Set kernel argument meta data.
    va_list args;
    va_start(args, narg);

    for (unsigned i = 0; i < narg; i++) {
      strncpy(prg->args[i].name, va_arg(args, const char *), MAX_ARG_NAME_SIZE);
      int type_and_attrs = va_arg(args, int);
      prg->args[i].size = va_arg(args, size_t);
      int attrs = type_and_attrs & NOMP_ATTR_MASK;
      prg->args[i].type = type_and_attrs - attrs;
      prg->args[i].pinned = attrs & NOMP_ATTR_PINNED;
      if (attrs & NOMP_ATTR_REDN) {
        if (prg->reduction_indx >= 0) {
          // FIXME: return set_log();
        }
        prg->reduction_indx = i;
      }
    }

    va_end(args);

    // Parse the clauses
    struct meta info;
    int err = parse_clauses(&info, clauses);
    return_on_err(err);

    // Create loopy kernel from C source.
    PyObject *knl = NULL;
    err = py_c_to_loopy(&knl, c_src, nomp.backend);
    return_on_err(err);

    char *name = NULL;
    err = py_get_knl_name(&name, knl);
    return_on_err(err);

    // Handle annotate clauses if they exist.
    err = py_user_annotate(&knl, info.dict, nomp.script, nomp.annts_func);
    return_on_err(err);

    // Handle transform clause.
    err = py_user_transform(&knl, info.file, info.func);
    return_on_err(err);

    // Handle reductions if present.
    if (prg->reduction_indx >= 0) {
      err = py_handle_reduction(&knl, &prg->reduction_op, nomp.backend);
      return_on_err(err);
    }

    // Get OpenCL, CUDA, etc. source from the loopy kernel.
    char *src = NULL;
    err = py_get_knl_src(&src, knl);
    return_on_err(err);

    // Build the kernel
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

    free_meta(&info);

    *id = progs_n++;
  }

  return 0;
}

int nomp_run(int id, ...) {
  if (id >= 0 && id < progs_n && progs[id] != NULL) {
    struct prog *prg = progs[id];
    struct arg *args = prg->args;

    va_list vargs;
    va_start(vargs, id);

    struct mem *m;
    for (int i = 0; i < prg->narg; i++) {
      void *arg = args[i].hptr = va_arg(vargs, void *);
      // Get meta data we stored during nomp_jit()
      const char *name = args[i].name;
      unsigned type = args[i].type;
      size_t size = args[i].size;

      PyObject *key, *val;
      switch (type) {
      case NOMP_INT:
      case NOMP_UINT:
      case NOMP_FLOAT:
        // FIXME: Different types should be handled differently.
        val = PyLong_FromLong(*((int *)arg));
        key = PyUnicode_FromStringAndSize(name, strlen(name));
        PyDict_SetItem(prg->py_dict, key, val);
        Py_XDECREF(key), Py_XDECREF(val);
        args[i].ptr = arg;
        break;
      case NOMP_PTR:
        m = mem_if_mapped(arg);
        if (m == NULL) {
          if (prg->reduction_indx != i) {
            return set_log(NOMP_USER_MAP_PTR_IS_INVALID, NOMP_ERROR,
                           ERR_STR_USER_MAP_PTR_IS_INVALID, arg);
          }
          m = nomp.buffer;
        }
        args[i].ptr = (void *)&m->bptr;
        break;
      default:
        return set_log(
            NOMP_USER_KNL_ARG_TYPE_IS_INVALID, NOMP_ERROR,
            "Kernel argument type %d passed to libnomp is not valid.", type);
        break;
      }
    }

    va_end(vargs);

    int err = py_eval_grid_size(prg, prg->py_dict);
    return_on_err(err);

    err = nomp.knl_run(&nomp, prg);
    return_on_err(err);

    if (prg->reduction_indx >= 0) {
      err = host_side_reduction(&nomp, prg, nomp.buffer);
      return_on_err(err);
    }

    return 0;
  }
  return set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
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
  if (!initialized) {
    return set_log(NOMP_RUNTIME_NOT_INITIALIZED, NOMP_ERROR,
                   "libnomp is not initialized.");
  }

  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i]) {
      // FIXME: Check error returned form `nomp.update`
      int err = nomp.update(&nomp, mems[i], NOMP_FREE);
      tfree(mems[i]), mems[i] = NULL;
    }
  }
  tfree(mems), mems = NULL, mems_n = mems_max = 0;

  for (unsigned i = 0; i < progs_n; i++) {
    if (progs[i]) {
      // FIXME: Check error returned form `nomp.knl_free`.
      // FIXME: Free program arguments.
      int err = nomp.knl_free(progs[i]);
      tfree(progs[i]), progs[i] = NULL;
    }
  }
  tfree(progs), progs = NULL, progs_n = progs_max = 0;

  // FIXME: Check error returned from `nomp_update`.
  int err = nomp.update(&nomp, nomp.buffer, NOMP_FREE);
  tfree(nomp.buffer->hptr);
  tfree(nomp.buffer), nomp.buffer = NULL;

  initialized = nomp.finalize(&nomp);
  if (initialized) {
    return set_log(NOMP_RUNTIME_FAILED_TO_FINALIZE, NOMP_ERROR,
                   "Failed to initialize libnomp.");
  }

  tfree(nomp.backend), nomp.backend = NULL;
  tfree(nomp.install_dir), nomp.install_dir = NULL;
  tfree(nomp.script), nomp.script = NULL;
  tfree(nomp.annts_func), nomp.annts_func = NULL;

  finalize_logs();

  return 0;
}
