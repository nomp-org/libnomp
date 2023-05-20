#include "nomp-impl.h"
#include "nomp-reduction.h"

static struct nomp_backend nomp;
static int initialized = 0;

static inline char *copy_env(const char *name, size_t size) {
  const char *tmp = getenv(name);
  if (tmp)
    return strndup(tmp, size);
  return NULL;
}

static int check_env(struct nomp_backend *backend) {
  char *tmp = getenv("NOMP_PLATFORM");
  if (tmp)
    backend->platform_id = nomp_str_toui(tmp, NOMP_MAX_BUFSIZ);

  if ((tmp = getenv("NOMP_DEVICE")))
    backend->device_id = nomp_str_toui(tmp, NOMP_MAX_BUFSIZ);

  if ((tmp = getenv("NOMP_VERBOSE")))
    backend->verbose = nomp_str_toui(tmp, NOMP_MAX_BUFSIZ);

  if ((tmp = getenv("NOMP_PROFILE")))
    backend->profile = nomp_str_toui(tmp, NOMP_MAX_BUFSIZ);

  if ((tmp = copy_env("NOMP_BACKEND", NOMP_MAX_BUFSIZ))) {
    nomp_free(&backend->backend);
    backend->backend = strndup(tmp, NOMP_MAX_BUFSIZ), nomp_free(&tmp);
  }

  if ((tmp = copy_env("NOMP_ANNOTATE_FUNCTION", NOMP_MAX_BUFSIZ))) {
    nomp_check(nomp_py_set_annotate_func(&backend->py_annotate, tmp));
    nomp_free(&tmp);
  }

  if ((tmp = copy_env("NOMP_INSTALL_DIR", NOMP_MAX_BUFSIZ))) {
    nomp_free(&backend->install_dir);
    size_t size;
    nomp_check(nomp_path_len(&size, tmp));
    backend->install_dir = strndup(tmp, size + 1), nomp_free(&tmp);
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

static int init_configs(int argc, const char **argv,
                        struct nomp_backend *backend) {
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
      if (!strncmp("--nomp-backend", argv[i], NOMP_MAX_BUFSIZ)) {
        backend->backend = strndup((const char *)argv[i + 1], NOMP_MAX_BUFSIZ);
      } else if (!strncmp("--nomp-platform", argv[i], NOMP_MAX_BUFSIZ)) {
        backend->platform_id = nomp_str_toui(argv[i + 1], NOMP_MAX_BUFSIZ);
      } else if (!strncmp("--nomp-device", argv[i], NOMP_MAX_BUFSIZ)) {
        backend->device_id = nomp_str_toui(argv[i + 1], NOMP_MAX_BUFSIZ);
      } else if (!strncmp("--nomp-verbose", argv[i], NOMP_MAX_BUFSIZ)) {
        backend->verbose = nomp_str_toui(argv[i + 1], NOMP_MAX_BUFSIZ);
      } else if (!strncmp("--nomp-profile", argv[i], NOMP_MAX_BUFSIZ)) {
        backend->profile = nomp_str_toui(argv[i + 1], NOMP_MAX_BUFSIZ);
      } else if (!strncmp("--nomp-install-dir", argv[i], NOMP_MAX_BUFSIZ)) {
        size_t size;
        nomp_check(nomp_path_len(&size, (const char *)argv[i + 1]));
        backend->install_dir = strndup((const char *)argv[i + 1], size + 1);
      } else if (!strncmp("--nomp-function", argv[i], NOMP_MAX_BUFSIZ)) {
        nomp_check(nomp_py_set_annotate_func(&backend->py_annotate,
                                             (const char *)argv[i + 1]));
      }
      i++;
    }
    i++;
  }

  nomp_check(check_env(&nomp));

#define check_if_initialized(param, dummy, cmd_arg, env_var)                   \
  {                                                                            \
    if (backend->param == (dummy)) {                                           \
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
  size_t len;
  nomp_check(nomp_path_len(&len, nomp.install_dir));
  len = nomp_max(2, len, NOMP_MAX_BUFSIZ);
  char *abs_dir = nomp_str_cat(2, len, nomp.install_dir, "/python");
  nomp_check(nomp_py_append_to_sys_path(abs_dir));
  nomp_free(&abs_dir);
  return 0;
}

static int allocate_scratch_memory(struct nomp_backend *backend) {
  if (backend->scratch == NULL) {
    struct nomp_mem *m = backend->scratch = nomp_calloc(struct nomp_mem, 1);
    m->idx0 = 0, m->idx1 = NOMP_MAX_SCRATCH_SIZE, m->usize = sizeof(double);
    m->hptr = nomp_calloc(double, m->idx1 - m->idx0);
    nomp_check(backend->update(backend, m, NOMP_ALLOC));
    return 0;
  }
  return nomp_set_log(NOMP_MEMORY_FAILURE, NOMP_ERROR,
                      "Unable to allocate scratch memory for kernels.");
}

static int deallocate_scratch_memory(struct nomp_backend *backend) {
  if (backend->scratch) {
    nomp_check(backend->update(backend, backend->scratch, NOMP_FREE));
    nomp_free(&backend->scratch->hptr), nomp_free(&backend->scratch);
    return 0;
  }
  return nomp_set_log(NOMP_MEMORY_FAILURE, NOMP_ERROR,
                      "Unable to deallocate scratch memory for kernels.");
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
    nomp_check(nomp_py_append_to_sys_path("."));
  }

  nomp_check(init_configs(argc, argv, &nomp));

  nomp_profile("nomp_init", 1, nomp.profile, 1);
  nomp_check(nomp_log_init(nomp.verbose));

  size_t n = strnlen(nomp.backend, NOMP_MAX_BUFSIZ);
  for (int i = 0; i < n; i++)
    nomp.backend[i] = tolower(nomp.backend[i]);

  if (strncmp(nomp.backend, "opencl", NOMP_MAX_BUFSIZ) == 0) {
#if defined(OPENCL_ENABLED)
    nomp_check(opencl_init(&nomp, nomp.platform_id, nomp.device_id));
#endif
  } else if (strncmp(nomp.backend, "cuda", NOMP_MAX_BUFSIZ) == 0) {
#if defined(CUDA_ENABLED)
    nomp_check(cuda_init(&nomp, nomp.platform_id, nomp.device_id));
#endif
  } else if (strncmp(nomp.backend, "hip", NOMP_MAX_BUFSIZ) == 0) {
#if defined(HIP_ENABLED)
    nomp_check(hip_init(&nomp, nomp.platform_id, nomp.device_id));
#endif
  } else if (strncmp(nomp.backend, "sycl", NOMP_MAX_BUFSIZ) == 0) {
#if defined(SYCL_ENABLED)
    nomp_check(sycl_init(&nomp, nomp.platform_id, nomp.device_id));
#endif
  } else if (strncmp(nomp.backend, "ispc", NOMP_MAX_BUFSIZ) == 0) {
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

  // Populate context
  nomp.py_context = PyDict_New();
  PyObject *pbackend = PyUnicode_FromString(nomp.backend);
  PyDict_SetItemString(nomp.py_context, "backend", pbackend);

  initialized = 1;
  nomp_profile("nomp_init", 0, nomp.profile, 1);
  return 0;
}

static struct nomp_mem **mems = NULL;
static int mems_n = 0;
static int mems_max = 0;

/**
 * @ingroup nomp_mem_utils
 * @brief Returns the nomp_mem object corresponding to host pointer \p p.
 *
 * Returns the nomp_mem object corresponding to host ponter \p p. If no buffer
 * has been allocated for \p p on the device, returns NULL.
 *
 * @param[in] p Host pointer
 * @return struct nomp_mem *
 */
static inline struct nomp_mem *mem_if_mapped(void *p) {
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
  nomp_profile("nomp_update", 1, nomp.profile, 1);
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
      mems = nomp_realloc(mems, struct nomp_mem *, mems_max);
    }
    struct nomp_mem *m = mems[mems_n] = nomp_calloc(struct nomp_mem, 1);
    m->idx0 = idx0, m->idx1 = idx1, m->usize = usize;
    m->hptr = ptr, m->bptr = NULL;
  }

  nomp_check(nomp.update(&nomp, mems[idx], op));

  // Device memory object was released.
  if (mems[idx]->bptr == NULL)
    nomp_free(&mems[idx]);
  // Or new memory object got created.
  else if (idx == mems_n)
    mems_n++;

  nomp_profile("nomp_update", 0, nomp.profile, 1);
  return 0;
}

static struct nomp_prog **progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

struct meta {
  char *file, *func;
  PyObject *dict;
};

static int parse_clauses(struct meta *meta, struct nomp_prog *prg,
                         const char **clauses) {
  // Currently, we only support `transform` and `annotate` and `jit`.
  meta->dict = PyDict_New(), meta->file = meta->func = NULL;
  unsigned i = 0;
  while (clauses[i]) {
    if (strncmp(clauses[i], "transform", NOMP_MAX_BUFSIZ) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return nomp_set_log(
            NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");
      }
      nomp_check(nomp_check_py_script_path((const char *)clauses[i + 1]));
      meta->file = strndup(clauses[i + 1], PATH_MAX);
      meta->func = strndup(clauses[i + 2], NOMP_MAX_BUFSIZ);
      i += 3;
    } else if (strncmp(clauses[i], "annotate", NOMP_MAX_BUFSIZ) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return nomp_set_log(
            NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
            "\"annotate\" clause should be followed by a key value "
            "pair. At least one of them is not provided.");
      }
      const char *key = clauses[i + 1], *val = clauses[i + 2];
      PyObject *pkey =
          PyUnicode_FromStringAndSize(key, strnlen(key, NOMP_MAX_BUFSIZ));
      PyObject *pval =
          PyUnicode_FromStringAndSize(val, strnlen(val, NOMP_MAX_BUFSIZ));
      PyDict_SetItem(meta->dict, pkey, pval);
      Py_XDECREF(pkey), Py_XDECREF(pval);
      i += 3;
    } else if (strncmp(clauses[i], "reduce", NOMP_MAX_BUFSIZ) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return nomp_set_log(
            NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
            "\"reduce\" clause should be followed by a variable name and an "
            "operation. At least one of them is not provided.");
      }
      for (unsigned j = 0; j < prg->nargs; j++) {
        if (strncmp(prg->args[j].name, clauses[i + 1], NOMP_MAX_BUFSIZ) == 0) {
          prg->reduction_type = prg->args[j].type, prg->args[j].type = NOMP_PTR;
          prg->reduction_index = j;
          break;
        }
      }
      if (strncmp(clauses[i + 2], "+", 2) == 0)
        prg->reduction_op = 0;
      if (strncmp(clauses[i + 2], "*", 2) == 0)
        prg->reduction_op = 1;
      i += 3;
    } else if (strncmp(clauses[i], "pin", NOMP_MAX_BUFSIZ) == 0) {
      // Check if we have to use pinned memory on the device.
      return nomp_set_log(NOMP_NOT_IMPLEMENTED_ERROR, NOMP_ERROR,
                          "Pinned memory support is not yet implemented.");
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
  nomp_profile("nomp_jit", 1, nomp.profile, 1);
  if (*id == -1) {
    if (progs_n == progs_max) {
      progs_max += progs_max / 2 + 1;
      progs = nomp_realloc(progs, struct nomp_prog *, progs_max);
    }

    // Initialize the program struct.
    struct nomp_prog *prg = progs[progs_n] = nomp_calloc(struct nomp_prog, 1);
    prg->nargs = nargs, prg->args = nomp_calloc(struct nomp_arg, nargs);
    prg->py_dict = PyDict_New(), prg->reduction_index = -1;

    va_list args;
    va_start(args, nargs);
    for (unsigned i = 0; i < prg->nargs; i++) {
      const char *name = va_arg(args, const char *);
      strncpy(prg->args[i].name, name, NOMP_MAX_BUFSIZ);
      prg->args[i].size = va_arg(args, size_t);
      prg->args[i].type = va_arg(args, int);
    }
    va_end(args);

    // Parse the clauses to find transformations file, function and other
    // annotations. Annotations are returned as a Python dictionary.
    struct meta meta;
    nomp_check(parse_clauses(&meta, prg, clauses));

    // Create loopy kernel from C source.
    PyObject *knl = NULL;
    nomp_check(nomp_py_c_to_loopy(&knl, csrc, nomp.backend));
    if (prg->reduction_index >= 0)
      nomp_check(nomp_py_realize_reduction(
          &knl, prg->args[prg->reduction_index].name));

    // Handle annotate clauses if they exist.
    nomp_check(nomp_py_apply_annotations(&knl, nomp.py_annotate, meta.dict,
                                         nomp.py_context));
    Py_XDECREF(meta.dict);

    // Handle transform clauses.
    nomp_check(
        nomp_py_apply_transform(&knl, meta.file, meta.func, nomp.py_context));
    nomp_free(&meta.file), nomp_free(&meta.func);

    // Get OpenCL, CUDA, etc. source and name from the loopy kernel and build
    // the program.
    char *name, *src;
    nomp_check(nomp_py_get_knl_name_and_src(&name, &src, knl, nomp.backend));
    nomp_check(nomp.knl_build(&nomp, prg, src, name));
    nomp_free(&src), nomp_free(&name);

    // Get grid size of the loopy kernel as pymbolic expressions.
    // These grid sizes will be evaluated each time the kernel is run.
    nomp_check(nomp_py_get_grid_size(prg, knl));
    Py_XDECREF(knl);

    *id = progs_n++;
  }

  nomp_profile("nomp_jit", 0, nomp.profile, 1);
  return 0;
}

int nomp_run(int id, ...) {
  if (id >= 0) {
    nomp_profile("nomp_run setup time", 1, nomp.profile, 1);
    struct nomp_prog *prg = progs[id];
    struct nomp_arg *args = prg->args;

    PyObject *key, *val;
    struct nomp_mem *m;

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
        if (prg->reduction_index == i) {
          prg->reduction_ptr = args[i].ptr, prg->reduction_size = args[i].size;
          m = nomp.scratch;
        } else {
          m = mem_if_mapped(args[i].ptr);
          if (m == NULL) {
            return nomp_set_log(NOMP_USER_MAP_PTR_IS_INVALID, NOMP_ERROR,
                                ERR_STR_USER_MAP_PTR_IS_INVALID, args[i].ptr);
          }
        }
        args[i].size = m->bsize, args[i].ptr = m->bptr;
        break;
      }
    }
    va_end(vargs);
    nomp_profile("nomp_run setup time", 0, nomp.profile, 1);

    nomp_profile("nomp_run grid evaluation", 1, nomp.profile, 1);
    nomp_check(nomp_py_eval_grid_size(prg));
    nomp_check(nomp.knl_run(&nomp, prg));
    if (prg->reduction_index >= 0)
      nomp_check(nomp_host_side_reduction(&nomp, prg, nomp.scratch));
    nomp_profile("nomp_run kernel runtime", 0, nomp.profile, 1);

    return 0;
  }

  return nomp_set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                      "Kernel id %d passed to nomp_run is not valid.", id);
}

int nomp_sync() { return nomp.sync(&nomp); }

int nomp_finalize(void) {
  nomp_profile("nomp_finalize", 1, nomp.profile, 1);
  if (!initialized) {
    return nomp_set_log(NOMP_FINALIZE_FAILURE, NOMP_ERROR,
                        "libnomp is not initialized.");
  }

  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i]) {
      nomp_check(nomp.update(&nomp, mems[i], NOMP_FREE));
      nomp_free(&mems[i]);
    }
  }
  nomp_free(&mems), mems_n = mems_max = 0;

  for (unsigned i = 0; i < progs_n; i++) {
    if (progs[i]) {
      nomp_check(nomp.knl_free(progs[i]));
      Py_XDECREF(progs[i]->py_global), Py_XDECREF(progs[i]->py_local);
      Py_XDECREF(progs[i]->py_dict), nomp_free(&progs[i]->args);
    }
    nomp_free(&progs[i]);
  }
  nomp_free(&progs), progs_n = progs_max = 0;

  nomp_check(deallocate_scratch_memory(&nomp));

  nomp_free(&nomp.backend), nomp_free(&nomp.install_dir);
  Py_XDECREF(nomp.py_annotate), Py_XDECREF(nomp.py_context);

  initialized = nomp.finalize(&nomp);
  if (initialized) {
    return nomp_set_log(NOMP_FINALIZE_FAILURE, NOMP_ERROR,
                        "Failed to initialize libnomp.");
  }
  nomp_profile("nomp_finalize", 0, nomp.profile, 0);
  if (nomp.profile == 1)
    nomp_profile_result();

  // Free all the logs
  nomp_logs_finalize();
  nomp_profile_finalize();

  return 0;
}
