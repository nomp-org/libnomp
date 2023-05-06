#include "nomp-impl.h"
#include "nomp-reduction.h"

static struct backend nomp;
static int initialized = 0;

static inline char *copy_env(const char *name, size_t size) {
  const char *tmp = getenv(name);
  if (tmp)
    return strndup(tmp, size);
  return NULL;
}

static int check_env(struct backend *backend) {
  char *tmp = getenv("NOMP_PLATFORM");
  if (tmp)
    backend->platform_id = nomp_str_toui(tmp, MAX_BUFSIZ);

  if ((tmp = getenv("NOMP_DEVICE")))
    backend->device_id = nomp_str_toui(tmp, MAX_BUFSIZ);

  if ((tmp = getenv("NOMP_VERBOSE")))
    backend->verbose = nomp_str_toui(tmp, MAX_BUFSIZ);

  if ((tmp = copy_env("NOMP_BACKEND", MAX_BACKEND_SIZE))) {
    if (backend->backend)
      nomp_free(backend->backend);
    backend->backend = strndup(tmp, MAX_BACKEND_SIZE), nomp_free(tmp);
  }

  if ((tmp = copy_env("NOMP_ANNOTATE_FUNCTION", MAX_BUFSIZ))) {
    nomp_check(py_set_annotate_func(&backend->py_annotate, tmp));
    nomp_free(tmp);
  }

  if ((tmp = copy_env("NOMP_INSTALL_DIR", MAX_BUFSIZ))) {
    size_t size;
    nomp_check(nomp_path_len(&size, tmp));
    if (backend->install_dir)
      nomp_free(backend->install_dir);
    backend->install_dir = strndup(tmp, size + 1), nomp_free(tmp);
  }

  return 0;
}

static inline int check_cmd_line_arg(unsigned i, unsigned argc,
                                     const char *argv[]) {
  if (i >= argc || argv[i] == NULL) {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "Missing argument value after: %s.", argv[i - 1]);
  }
  return 0;
}

static int init_configs(int argc, const char **argv, struct backend *backend) {
  // We only a provide default value for verbose. Everything else has to be set
  // by user explicitly.
  backend->verbose = 0;
  backend->device_id = backend->platform_id = -1;
  backend->backend = backend->install_dir = NULL;

  if (argc <= 1 || argv == NULL)
    return 0;

  unsigned i = 0;
  while (i < argc) {
    if (!strncmp("--nomp", argv[i], 6)) {
      nomp_check(check_cmd_line_arg(i + 1, argc, argv));
      if (!strncmp("--nomp-backend", argv[i], MAX_BUFSIZ)) {
        backend->backend = strndup((const char *)argv[i + 1], MAX_BACKEND_SIZE);
      } else if (!strncmp("--nomp-platform", argv[i], MAX_BUFSIZ)) {
        backend->platform_id = nomp_str_toui(argv[i + 1], MAX_BUFSIZ);
      } else if (!strncmp("--nomp-device", argv[i], MAX_BUFSIZ)) {
        backend->device_id = nomp_str_toui(argv[i + 1], MAX_BUFSIZ);
      } else if (!strncmp("--nomp-verbose", argv[i], MAX_BUFSIZ)) {
        backend->verbose = nomp_str_toui(argv[i + 1], MAX_BUFSIZ);
      } else if (!strncmp("--nomp-install-dir", argv[i], MAX_BUFSIZ)) {
        size_t size;
        nomp_check(nomp_path_len(&size, (const char *)argv[i + 1]));
        backend->install_dir = strndup((const char *)argv[i + 1], size + 1);
      } else if (!strncmp("--nomp-function", argv[i], MAX_BUFSIZ)) {
        nomp_check(py_set_annotate_func(&backend->py_annotate,
                                        (const char *)argv[i + 1]));
      }
      i++;
    }
    i++;
  }

  nomp_check(check_env(&nomp));

#define check_if_initialized(param, dummy, cmd_arg, env_var)                   \
  {                                                                            \
    if (backend->param == dummy) {                                             \
      return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,              \
                          #param                                               \
                          " is missing or invalid. Set it with " cmd_arg       \
                          " command line argument or " env_var                 \
                          " environment variable.");                           \
    }                                                                          \
  }

  check_if_initialized(device_id, -1, "--nomp-device", "NOMP_DEVICE");
  check_if_initialized(platform_id, -1, "--nomp-platform", "NOMP_PLATFORM");
  check_if_initialized(backend, NULL, "--nomp-backend", "NOMP_BACKEND");
  check_if_initialized(install_dir, NULL, "--nomp-install-dir",
                       "NOMP_INSTALL_DIR");

#undef check_if_initialized

  // Append nomp python directory to sys.path.
  // nomp.install_dir should be set and we use it here.
  size_t len;
  nomp_check(nomp_path_len(&len, nomp.install_dir));
  len = nomp_max(2, len, MAX_BUFSIZ);
  char *abs_dir = nomp_str_cat(2, len, nomp.install_dir, "/python");
  nomp_check(py_append_to_sys_path(abs_dir));
  nomp_free(abs_dir);

  return 0;
}

static int allocate_scratch_memory(struct backend *backend) {
  if (backend->scratch == NULL) {
    struct mem *m = backend->scratch = nomp_calloc(struct mem, 1);
    m->idx0 = 0, m->idx1 = MAX_SCRATCH_SIZE, m->usize = sizeof(char);
    m->hptr = nomp_calloc(double, m->idx1 - m->idx0);
    nomp_check(backend->update(backend, m, NOMP_ALLOC));
    return 0;
  }
  // FIXME: Return an error.
  return 0;
}

static int deallocate_scratch_memory(struct backend *backend) {
  if (backend->scratch) {
    nomp_check(backend->update(backend, backend->scratch, NOMP_FREE));
    nomp_free(backend->scratch->hptr);
    nomp_free(backend->scratch);
    return 0;
  }
  // FIXME: Return an error.
  return 0;
}

int nomp_init(int argc, const char **argv) {
  if (initialized) {
    return nomp_set_log(NOMP_INITIALIZE_FAILURE, NOMP_ERROR,
                        "libnomp is already initialized.");
  }

  if (!Py_IsInitialized()) {
    // May be we need the isolated configuration listed here:
    // https://docs.python.org/3/c-api/init_config.html#init-config
    // But for now, we do the simplest thing possible.
    Py_Initialize();

    // Append current working directory to sys.path.
    nomp_check(py_append_to_sys_path("."));
  }

  nomp_check(init_configs(argc, argv, &nomp));

  nomp_check(nomp_log_init(nomp.verbose));

  size_t n = strnlen(nomp.backend, MAX_BACKEND_SIZE);
  for (int i = 0; i < n; i++)
    nomp.backend[i] = tolower(nomp.backend[i]);

  if (strncmp(nomp.backend, "opencl", MAX_BACKEND_SIZE) == 0) {
#if defined(OPENCL_ENABLED)
    nomp_check(opencl_init(&nomp, nomp.platform_id, nomp.device_id));
#endif
  } else if (strncmp(nomp.backend, "cuda", MAX_BACKEND_SIZE) == 0) {
#if defined(CUDA_ENABLED)
    nomp_check(cuda_init(&nomp, nomp.platform_id, nomp.device_id));
#endif
  } else if (strncmp(nomp.backend, "hip", MAX_BACKEND_SIZE) == 0) {
#if defined(HIP_ENABLED)
    nomp_check(hip_init(&nomp, nomp.platform_id, nomp.device_id));
#endif
  } else if (strncmp(nomp.backend, "sycl", MAX_BACKEND_SIZE) == 0) {
#if defined(SYCL_ENABLED)
    nomp_check(sycl_init(&nomp, nomp.platform_id, nomp.device_id));
#endif
  } else if (strncmp(nomp.backend, "ispc", MAX_BACKEND_SIZE) == 0) {
#if defined(ISPC_ENABLED)
    nomp_check(ispc_init(&nomp, nomp.platform_id, nomp.device_id));
#endif
  } else {
    return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "Failed to initialize libnomp. Invalid backend: %s",
                        nomp.backend);
  }

  nomp.scratch = NULL;
  nomp_check(allocate_scratch_memory(&nomp));

  initialized = 1;
  return 0;
}

static struct mem **mems = NULL;
static int mems_n = 0;
static int mems_max = 0;

/**
 * @ingroup nomp_mem_utils
 * @brief Returns the mem object corresponding to host pointer \p p.
 *
 * Returns the mem object corresponding to host ponter \p p. If no buffer has
 * been allocated for \p p on the device, returns NULL.
 *
 * @param[in] p Host pointer
 * @return struct mem *
 */
static inline struct mem *mem_if_mapped(void *p) {
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
  // Needs to go. Must store a hash map.
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
    // A new entry can't be created with NOMP_FREE or NOMP_FROM.
    if (op == NOMP_FROM || op == NOMP_FREE) {
      return nomp_set_log(
          NOMP_USER_MAP_OP_IS_INVALID, NOMP_ERROR,
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

  nomp_check(nomp.update(&nomp, mems[idx], op));

  // Device memory object was released.
  if (mems[idx]->bptr == NULL)
    nomp_free(mems[idx]), mems[idx] = NULL;
  // Or new memory object got created.
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
        return nomp_set_log(
            NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");
      }
      nomp_check(nomp_check_py_script_path((const char *)clauses[i + 1]));
      *usr_file = strndup(clauses[i + 1], PATH_MAX);
      *usr_func = strndup(clauses[i + 2], MAX_FUNC_NAME_SIZE);
      i += 3;
    } else if (strncmp(clauses[i], "annotate", MAX_BUFSIZ) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return nomp_set_log(
            NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
            "\"annotate\" clause should be followed by a key value "
            "pair. At least one of them is not provided.");
      }
      const char *key = clauses[i + 1], *val = clauses[i + 2];
      PyObject *pkey =
          PyUnicode_FromStringAndSize(key, strnlen(key, MAX_KEY_SIZE));
      PyObject *pval =
          PyUnicode_FromStringAndSize(val, strnlen(val, MAX_VAL_SIZE));
      PyDict_SetItem(dict, pkey, pval);
      Py_XDECREF(pkey), Py_XDECREF(pval);
      i += 3;
    } else {
      return nomp_set_log(
          NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
          "Clause \"%s\" passed into nomp_jit is not a valid clause.",
          clauses[i]);
    }
  }

  return 0;
}

int nomp_jit(int *id, const char *csrc, const char **clauses, int nargs, ...) {
  if (*id == -1) {
    if (progs_n == progs_max) {
      progs_max += progs_max / 2 + 1;
      progs = nomp_realloc(progs, struct prog *, progs_max);
    }

    struct prog *prg = progs[progs_n] = nomp_calloc(struct prog, 1);
    prg->py_dict = PyDict_New(), prg->reduction_index = -1;
    prg->nargs = nargs, prg->args = nomp_calloc(struct arg, nargs);

    va_list args;
    va_start(args, nargs);
    for (unsigned i = 0; i < prg->nargs; i++) {
      const char *name = va_arg(args, const char *);
      strncpy(prg->args[i].name, name, MAX_ARG_NAME_SIZE);
      prg->args[i].size = va_arg(args, size_t);
      int type_and_attrs = va_arg(args, int);
      prg->args[i].type = type_and_attrs & NOMP_ATTRIBUTE_MASK;
      // Check if the argument is part of a reduction.
      if (type_and_attrs & NOMP_ATTRIBUTE_REDUCTION) {
        // Check if there was a reduction in the kernel already and bail
        // out if that is the case.
        if (prg->reduction_index >= 0) {
          return nomp_set_log(
              NOMP_NOT_IMPLEMENTED_ERROR, NOMP_ERROR,
              "Multiple reductions in a kernel is not yet implemented.");
        }
        prg->reduction_index = i, prg->reduction_type = prg->args[i].type;
        prg->args[i].type = NOMP_PTR;
      }
      // Check if we have to use pinned memory on the device.
      if (type_and_attrs & NOMP_ATTRIBUTE_PINNED) {
        return nomp_set_log(NOMP_NOT_IMPLEMENTED_ERROR, NOMP_ERROR,
                            "Pinned memory support is not yet implemented.");
      }
    }
    va_end(args);

    // Create loopy kernel from C source.
    PyObject *knl = NULL;
    nomp_check(py_c_to_loopy(&knl, &prg->reduction_op, csrc, nomp.backend,
                             prg->reduction_index));

    // Parse the clauses to find transformations file, function and other
    // annotations. Annotations are returned as a Python dictionary.
    char *file = NULL, *func = NULL;
    PyObject *annotations = NULL;
    nomp_check(parse_clauses(&file, &func, &annotations, clauses));

    // Handle annotate clauses if they exist.
    nomp_check(py_apply_annotations(&knl, nomp.py_annotate, annotations));
    Py_XDECREF(annotations);

    // Handle transform clauses.
    nomp_check(py_user_transform(&knl, file, func));
    nomp_free(file), nomp_free(func);

    // Get OpenCL, CUDA, etc. source and name from the loopy kernel
    char *name, *src;
    nomp_check(py_get_knl_name_and_src(&name, &src, knl, nomp.backend));

    // Build the kernel
    nomp_check(nomp.knl_build(&nomp, prg, src, name));
    nomp_free(src), nomp_free(name);

    // Get grid size of the loopy kernel as pymbolic expressions.
    // These grid sizes will be evaluated each time the kernel is run.
    nomp_check(py_get_grid_size(prg, knl));
    Py_XDECREF(knl);

    *id = progs_n++;
  }

  return 0;
}

int nomp_run(int id, ...) {
  if (id >= 0) {
    struct prog *prg = progs[id];
    struct arg *args = prg->args;

    PyObject *key, *val;
    struct mem *m;

    va_list vargs;
    va_start(vargs, id);
    for (unsigned i = 0; i < prg->nargs; i++) {
      args[i].ptr = va_arg(vargs, void *);
      switch (args[i].type) {
      case NOMP_INT:
        val = PyLong_FromLong(*((int *)args[i].ptr));
        goto key;
        break;
      case NOMP_UINT:
        val = PyLong_FromLong(*((unsigned int *)args[i].ptr));
      key:
        key = PyUnicode_FromStringAndSize(args[i].name, strlen(args[i].name));
        PyDict_SetItem(prg->py_dict, key, val);
        Py_XDECREF(key), Py_XDECREF(val);
        break;
      case NOMP_PTR:
        m = mem_if_mapped(args[i].ptr);
        if (m == NULL) {
          if (prg->reduction_index == i) {
            prg->reduction_ptr = args[i].ptr,
            prg->reduction_size = args[i].size;
            m = nomp.scratch;
          } else {
            return nomp_set_log(NOMP_USER_MAP_PTR_IS_INVALID, NOMP_ERROR,
                                ERR_STR_USER_MAP_PTR_IS_INVALID, args[i].ptr);
          }
        }
        args[i].size = m->bsize, args[i].ptr = m->bptr;
        break;
      }
    }
    va_end(vargs);

    nomp_check(py_eval_grid_size(prg));

    // FIXME: Our kernel doesn't have the local problem size for some
    // reason.
    if (prg->reduction_index >= 0)
      prg->local[0] = 32;

    nomp_check(nomp.knl_run(&nomp, prg));

    if (prg->reduction_index >= 0) {
      nomp_sync();
      nomp_check(host_side_reduction(&nomp, prg, nomp.scratch));
    }

    return 0;
  }

  return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                      "Kernel id %d passed to nomp_run is not valid.", id);
}

int nomp_sync() { return nomp.sync(&nomp); }

int nomp_finalize(void) {
  if (!initialized) {
    return nomp_set_log(NOMP_FINALIZE_FAILURE, NOMP_ERROR,
                        "libnomp is not initialized.");
  }

  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i]) {
      nomp_check(nomp.update(&nomp, mems[i], NOMP_FREE));
      nomp_free(mems[i]), mems[i] = NULL;
    }
  }
  nomp_free(mems), mems = NULL, mems_n = mems_max = 0;

  for (unsigned i = 0; i < progs_n; i++) {
    if (progs[i]) {
      nomp_check(nomp.knl_free(progs[i]));
      nomp_free(progs[i]->args), nomp_free(progs[i]);
      Py_XDECREF(progs[i]->py_global), Py_XDECREF(progs[i]->py_local);
      Py_XDECREF(progs[i]->py_dict), progs[i] = NULL;
    }
  }
  nomp_free(progs), progs = NULL, progs_n = progs_max = 0;

  nomp_check(deallocate_scratch_memory(&nomp));

  nomp_free(nomp.backend), nomp_free(nomp.install_dir);
  Py_XDECREF(nomp.py_annotate);

  initialized = nomp.finalize(&nomp);
  if (initialized) {
    return nomp_set_log(NOMP_FINALIZE_FAILURE, NOMP_ERROR,
                        "Failed to initialize libnomp.");
  }

  return 0;
}
