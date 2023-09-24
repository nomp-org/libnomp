#include "nomp-impl.h"
#include "nomp-reduction.h"

static struct nomp_backend_t nomp;
static int initialized = 0;

static inline int check_env_vars(struct nomp_config_t *const cfg) {
  char *tmp = NULL;

  if ((tmp = getenv("NOMP_INSTALL_DIR")))
    strncpy(cfg->install_dir, tmp, PATH_MAX);

  if ((tmp = getenv("NOMP_BACKEND")))
    strncpy(cfg->backend, tmp, NOMP_MAX_BUFFER_SIZE);

  if ((tmp = getenv("NOMP_PLATFORM")))
    cfg->platform = nomp_str_toui(tmp, NOMP_MAX_BUFFER_SIZE);

  if ((tmp = getenv("NOMP_DEVICE")))
    cfg->device = nomp_str_toui(tmp, NOMP_MAX_BUFFER_SIZE);

  if ((tmp = getenv("NOMP_VERBOSE")))
    cfg->verbose = nomp_str_toui(tmp, NOMP_MAX_BUFFER_SIZE);

  if ((tmp = getenv("NOMP_PROFILE")))
    cfg->profile = nomp_str_toui(tmp, NOMP_MAX_BUFFER_SIZE);

  if ((tmp = getenv("NOMP_SCRIPTS_DIR")))
    strncpy(cfg->scripts_dir, tmp, PATH_MAX);

  return 0;
}

static inline int nomp_check_cmd_line_aux(const unsigned i, const unsigned argc,
                                          const char *const argv[]) {
  if (i >= argc || argv[i] == NULL) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Missing argument value after: %s.", argv[i - 1]);
  }
  return 0;
}

static inline int nomp_check_cmd_line(struct nomp_config_t *const cfg,
                                      unsigned argc, const char **argv) {
  if (argc <= 1 || argv == NULL)
    return 0;

  for (unsigned i = 0; i < argc;) {
    if (strncmp("--nomp", argv[i++], 6))
      continue;

    nomp_check(nomp_check_cmd_line_aux(i, argc, argv));

    int valid = 0;

    if (!strncmp("--nomp-install-dir", argv[i - 1], NOMP_MAX_BUFFER_SIZE))
      strncpy(cfg->install_dir, argv[i], PATH_MAX), valid = 1;

    if (!strncmp("--nomp-backend", argv[i - 1], NOMP_MAX_BUFFER_SIZE))
      strncpy(cfg->backend, argv[i], NOMP_MAX_BUFFER_SIZE), valid = 1;

    if (!strncmp("--nomp-platform", argv[i - 1], NOMP_MAX_BUFFER_SIZE)) {
      cfg->platform = nomp_str_toui(argv[i], NOMP_MAX_BUFFER_SIZE);
      valid = 1;
    }

    if (!strncmp("--nomp-device", argv[i - 1], NOMP_MAX_BUFFER_SIZE))
      cfg->device = nomp_str_toui(argv[i], NOMP_MAX_BUFFER_SIZE), valid = 1;

    if (!strncmp("--nomp-verbose", argv[i - 1], NOMP_MAX_BUFFER_SIZE))
      cfg->verbose = nomp_str_toui(argv[i], NOMP_MAX_BUFFER_SIZE), valid = 1;

    if (!strncmp("--nomp-profile", argv[i - 1], NOMP_MAX_BUFFER_SIZE))
      cfg->profile = nomp_str_toui(argv[i], NOMP_MAX_BUFFER_SIZE), valid = 1;

    if (!strncmp("--nomp-scripts-dir", argv[i - 1], NOMP_MAX_BUFFER_SIZE))
      strncpy(cfg->scripts_dir, argv[i], PATH_MAX), valid = 1;

    if (!valid) {
      nomp_log(0, NOMP_WARNING, "Unknown command line argument: %s.",
               argv[i - 1]);
    }

    i++;
  }

  return 0;
}

static inline int nomp_set_configs(int argc, const char **argv,
                                   struct nomp_config_t *const cfg) {
  // verbose, profile, device and platform id are all initialized to zero.
  // Everything else has to be set by user explicitly.
  cfg->verbose = NOMP_DEFAULT_VERBOSE;
  cfg->profile = NOMP_DEFAULT_PROFILE;
  cfg->device = NOMP_DEFAULT_DEVICE;
  cfg->platform = NOMP_DEFAULT_PLATFORM;
  strcpy(cfg->backend, "");
  strcpy(cfg->install_dir, "");
  strcpy(cfg->scripts_dir, "");

  nomp_check(nomp_check_cmd_line(cfg, argc, argv));
  check_env_vars(cfg);

  size_t n = strnlen(cfg->backend, NOMP_MAX_BUFFER_SIZE);
  for (unsigned i = 0; i < n; i++)
    cfg->backend[i] = tolower(cfg->backend[i]);

#define check_if_valid(COND, CMDARG, ENVVAR)                                   \
  {                                                                            \
    if (COND) {                                                                \
      return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,                  \
                      #ENVVAR " is missing or invalid. Set it with " CMDARG    \
                              " command line argument or " ENVVAR              \
                              " environment variable.");                       \
    }                                                                          \
  }

  check_if_valid(strlen(cfg->install_dir) == 0, "--nomp-install-dir",
                 "NOMP_INSTALL_DIR");
  check_if_valid(strlen(cfg->backend) == 0, "--nomp-backend", "NOMP_BACKEND");
  check_if_valid(cfg->verbose < 0, "--nomp-verbose", "NOMP_VERBOSE");
  check_if_valid(cfg->profile < 0, "--nomp-profile", "NOMP_PROFILE");
  check_if_valid(cfg->device < 0, "--nomp-device", "NOMP_DEVICE");
  check_if_valid(cfg->platform < 0, "--nomp-platform", "NOMP_PLATFORM");

#undef check_if_valid

  return 0;
}

static inline int allocate_scratch_memory(struct nomp_backend_t *bnd) {
  struct nomp_mem_t *m = &bnd->scratch;
  m->idx0 = 0, m->idx1 = NOMP_MAX_SCRATCH_SIZE, m->usize = sizeof(char);
  nomp_check(bnd->update(bnd, m, NOMP_ALLOC, m->idx0, m->idx1, m->usize));
  m->hptr = nomp_calloc(char, m->idx1 - m->idx0);
  return 0;
}

static inline int deallocate_scratch_memory(struct nomp_backend_t *bnd) {
  struct nomp_mem_t *m = &bnd->scratch;
  nomp_check(bnd->update(bnd, m, NOMP_FREE, m->idx0, m->idx1, m->usize));
  nomp_free(&m->hptr);
  return 0;
}

static inline int init_backend(struct nomp_backend_t *const bnd,
                               const struct nomp_config_t *const cfg) {
  bnd->py_context = PyDict_New();
  PyObject *obj = PyUnicode_FromString(cfg->backend);
  PyDict_SetItemString(bnd->py_context, "backend::name", obj);
  Py_XDECREF(obj);

  if (strncmp(cfg->backend, "opencl", NOMP_MAX_BUFFER_SIZE) == 0) {
#if defined(OPENCL_ENABLED)
    nomp_check(opencl_init(bnd, cfg->platform, cfg->device));
#endif
  } else if (strncmp(cfg->backend, "cuda", NOMP_MAX_BUFFER_SIZE) == 0) {
#if defined(CUDA_ENABLED)
    nomp_check(cuda_init(bnd, cfg->platform, cfg->device));
#endif
  } else if (strncmp(cfg->backend, "hip", NOMP_MAX_BUFFER_SIZE) == 0) {
#if defined(HIP_ENABLED)
    nomp_check(hip_init(bnd, cfg->platform, cfg->device));
#endif
  } else {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Invalid backend: %s.", cfg->backend);
  }

  return 0;
}

/**
 * @ingroup nomp_user_api
 *
 * @brief Initializes libnomp with the specified backend, platform, device, etc.
 *
 * @details Initializes nomp code generation for the specified backend (e.g.,
 * OpenCL, CUDA, etc) using command line arguments. Have a look at the accepted
 * arguments below for all available options. If no arguments are specified,
 * default values configured at build time are used for all configurations
 * except for backend and the install directory. The backend name and the
 * install directory are mandatory arguments.
 *
 * This function Returns a non-zero value if an error occurs during the
 * initialization, otherwise returns 0. Errors can be queried using
 * nomp_get_err_no() and nomp_get_err_str(). Calling this method multiple times
 * (without nomp_finalize in between) will return an error (but not segfault).
 *
 * <b>Accepted arguments:</b>
 * \arg `--nomp-install-dir <install-dir>` Specify `libnomp` install directory.
 * \arg `--nomp-backend <backend-name>` Specify backend name.
 * \arg `--nomp-platform <platform-index>` Specify platform id.
 * \arg `--nomp-device <device-index>` Specify device id.
 * \arg `--nomp-verbose <verbose-level>` Specify verbose level.
 * \arg `--nomp-profile <profile-level>` Specify profile level.
 * \arg `--nomp-scripts-dir <scripts-dir>` Specify the directory containing
 * annotation and transformation scripts.
 *
 * @param[in] argc The number of arguments to nomp_init().
 * @param[in] argv Arguments as strings, values followed by options.
 * @return int
 *
 * <b>Example usage:</b>
 * @code{.c}
 * const char *argv[] = {"--nomp-backend", "opencl", "--nomp-device", "0",
 * "--nomp-platform", "0"};
 * int argc = 6;
 * int err = nomp_init(argc, argv);
 * @endcode
 */
int nomp_init(int argc, const char **argv) {
  // This will be overridden by the user specified verbose level later.
  nomp_log_set_verbose(NOMP_DEFAULT_VERBOSE);

  if (initialized) {
    return nomp_log(NOMP_INITIALIZE_FAILURE, NOMP_ERROR,
                    "libnomp is already initialized.");
  }

  struct nomp_config_t cfg;
  nomp_check(nomp_set_configs(argc, argv, &cfg));

  nomp_check(nomp_py_init(&cfg));

  // Set profile level.
  nomp_check(nomp_profile_set_level(cfg.profile));

  // Set verbose level.
  nomp_check(nomp_log_set_verbose(cfg.verbose));

  // Initialize the backend.
  nomp_check(init_backend(&nomp, &cfg));

  // Allocate scratch memory.
  nomp_check(allocate_scratch_memory(&nomp));

  initialized = 1;

  nomp_log(0, NOMP_INFO, "libnomp initialized successfully.");

  return 0;
}

static struct nomp_mem_t **mems = NULL;
static unsigned mems_n = 0;
static unsigned mems_max = 0;

/**
 * @ingroup nomp_mem_utils
 * @brief Returns the nomp_mem object corresponding to host pointer \p p.
 *
 * Returns the nomp_mem object corresponding to host ponter \p p. If no buffer
 * has been allocated for \p p on the device, returns NULL.
 *
 * @param[in] p Host pointer
 * @return struct nomp_mem_t *
 */
static inline struct nomp_mem_t *mem_if_mapped(void *p) {
  // FIXME: This is O(N) in number of allocations.
  // Needs to go. Must store a hashmap.
  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i] && mems[i]->hptr == p)
      return mems[i];
  }
  return NULL;
}

static unsigned mem_if_exist(void *p, size_t idx0, size_t idx1, size_t usize) {
  // FIXME: This is O(N) in number of allocations.
  // Needs to go. Must store a hash map.
  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i] && mems[i]->hptr == p &&
        (mems[i]->idx0 * mems[i]->usize <= idx0 * usize) &&
        (mems[i]->idx1 * mems[i]->usize >= idx1 * usize))
      return i;
  }
  return mems_n;
}

/**
 * @ingroup nomp_user_api
 * @brief Performs device to host (D2H) and host to device (H2D) memory
 * transfers, allocating and freeing of memory in the device.
 *
 * @param[in] ptr Pointer to the host memory location.
 * @param[in] idx0 Start index in the vector to start copying.
 * @param[in] idx1 End index in the vector to end the copying.
 * @param[in] usize Size of a single vector element.
 * @param[in] op Operation to perform (One of #nomp_map_direction_t).
 * @return int
 *
 * @details Operation \p op will be performed on the array slice [\p start_idx,
 * \p end_idx), i.e., on array elements start_idx, ... end_idx - 1. This method
 * returns a non-zero value if there is an error and 0 otherwise.
 *
 * <b>Example usage:</b>
 * @code{.c}
 * int N = 10;
 * double a[10];
 * for (unsigned i = 0; i < N; i++)
 *   a[i] = i;
 *
 * // Copy the value of `a` into device
 * int err = nomp_update(a, 0, N, sizeof(double), NOMP_TO);
 *
 * // Execution of a kernel which uses `a`
 * ...
 *
 * // Copy the updated value of `a` from device
 * int err = nomp_update(a, 0, N, sizeof(double), NOMP_FROM);
 *
 * // Free the device memory allocated for `a`
 * int err = nomp_update(a, 0, N, sizeof(double), NOMP_FREE);
 *
 * @endcode
 */
int nomp_update(void *ptr, size_t idx0, size_t idx1, size_t usize,
                nomp_map_direction_t op) {
  unsigned idx = mem_if_exist(ptr, idx0, idx1, usize);
  if (idx == mems_n) {
    // A new entry can't be created with NOMP_FREE or
    // NOMP_FROM.
    if (op == NOMP_FROM || op == NOMP_FREE) {
      return nomp_log(NOMP_USER_MAP_OP_IS_INVALID, NOMP_ERROR,
                      "NOMP_FREE or NOMP_FROM can only be called "
                      "on a pointer "
                      "which is already on the device.");
    }
    op |= NOMP_ALLOC;
    if (mems_n == mems_max) {
      mems_max += mems_max / 2 + 1;
      mems = nomp_realloc(mems, struct nomp_mem_t *, mems_max);
    }
    struct nomp_mem_t *m = mems[mems_n] = nomp_calloc(struct nomp_mem_t, 1);
    m->idx0 = idx0, m->idx1 = idx1, m->usize = usize;
    m->hptr = ptr, m->bptr = NULL;
  }

  nomp_check(nomp.update(&nomp, mems[idx], op, idx0, idx1, usize));

  // Device memory object was released.
  if (mems[idx]->bptr == NULL)
    nomp_free(&mems[idx]);
  // Or new memory object got created.
  else if (idx == mems_n)
    mems_n++;

  return 0;
}

static struct nomp_prog_t **progs = NULL;
static unsigned progs_n = 0;
static unsigned progs_max = 0;

struct nomp_meta_t {
  char *file, *func;
  PyObject *dict;
};

static int parse_clauses(struct nomp_meta_t *meta, struct nomp_prog_t *prg,
                         const char **clauses) {
  // Currently, we only support `transform` and
  // `annotate` and `jit`.
  meta->dict = PyDict_New(), meta->file = meta->func = NULL;
  unsigned i = 0;
  while (clauses[i]) {
    if (strncmp(clauses[i], "transform", NOMP_MAX_BUFFER_SIZE) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "\"transform\" clause should be followed "
                        "by a file name and a "
                        "function name. At least one of them is "
                        "not provided.");
      }
      nomp_check(nomp_py_check_module((const char *)clauses[i + 1],
                                      (const char *)clauses[i + 2]));
      meta->file = strndup(clauses[i + 1], PATH_MAX);
      meta->func = strndup(clauses[i + 2], NOMP_MAX_BUFFER_SIZE);
      i += 3;
    } else if (strncmp(clauses[i], "annotate", NOMP_MAX_BUFFER_SIZE) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "\"annotate\" clause should be followed by "
                        "a key value "
                        "pair. At least one of them is not "
                        "provided.");
      }
      const char *key = clauses[i + 1], *val = clauses[i + 2];
      PyObject *pkey =
          PyUnicode_FromStringAndSize(key, strnlen(key, NOMP_MAX_BUFFER_SIZE));
      PyObject *pval =
          PyUnicode_FromStringAndSize(val, strnlen(val, NOMP_MAX_BUFFER_SIZE));
      PyDict_SetItem(meta->dict, pkey, pval);
      Py_XDECREF(pkey), Py_XDECREF(pval);
      i += 3;
    } else if (strncmp(clauses[i], "reduce", NOMP_MAX_BUFFER_SIZE) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "\"reduce\" clause should be followed by a "
                        "variable name and an "
                        "operation. At least one of them is not "
                        "provided.");
      }
      for (unsigned j = 0; j < prg->nargs; j++) {
        if (strncmp(prg->args[j].name, clauses[i + 1], NOMP_MAX_BUFFER_SIZE) ==
            0) {
          prg->redn_type = prg->args[j].type, prg->args[j].type = NOMP_PTR;
          prg->redn_size = prg->args[j].size, prg->redn_idx = j;
          break;
        }
      }
      if (strncmp(clauses[i + 2], "+", 2) == 0)
        prg->redn_op = NOMP_SUM;
      else if (strncmp(clauses[i + 2], "*", 2) == 0)
        prg->redn_op = NOMP_PROD;
      i += 3;
    } else if (strncmp(clauses[i], "pin", NOMP_MAX_BUFFER_SIZE) == 0) {
      // Check if we have to use pinned memory on the
      // device.
      return nomp_log(NOMP_NOT_IMPLEMENTED_ERROR, NOMP_ERROR,
                      "Pinned memory support is "
                      "not yet implemented.");
    } else {
      return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                      "Clause \"%s\" passed into nomp_jit is not a "
                      "valid clause.",
                      clauses[i]);
    }
  }

  return 0;
}

static inline struct nomp_prog_t *init_args(int progs_n, int nargs,
                                            va_list args) {
  struct nomp_prog_t *prg = progs[progs_n] = nomp_calloc(struct nomp_prog_t, 1);
  prg->args = nomp_calloc(struct nomp_arg_t, nargs);
  prg->nargs = nargs, prg->redn_idx = -1;
  prg->map = mapbasicbasic_new();
  prg->sym_global = vecbasic_new(), prg->sym_local = vecbasic_new();

  for (unsigned i = 0; i < prg->nargs; i++) {
    strncpy(prg->args[i].name, va_arg(args, const char *),
            NOMP_MAX_BUFFER_SIZE);
    prg->args[i].size = va_arg(args, size_t);
    prg->args[i].type = va_arg(args, int);
  }
  return prg;
}

/**
 * @ingroup nomp_user_api
 * @brief Generate and compile a kernel for the target backend (OpenCL, etc.)
 * from C source.
 *
 * @details Target backend is the one provided during the initialization of
 * libnomp using nomp_init(). User defined code transformations will be applied
 * based on the clauses specified in \p clauses argument. Additional kernel meta
 * data can be passed using the \p clauses as well. After \p clauses, number of
 * arguments to the kernel must be provided. Then for each argument, three
 * values has to be passed. First is the argument name as a string. Second is
 * is the `sizeof` argument and the third if argument type (one of @ref
 * nomp_user_types).
 *
 * <b>Example usage:</b>
 * @code{.c}
 * int N = 10;
 * double a[10], b[10];
 * for (unsigned i = 0; i < N; i++) {
 *   a[i] = i;
 *   b[i] = 10 -i
 * }
 * const char *knl = "for (unsigned i = 0; i < N; i++) a[i] += b[i];"
 * static int id = -1;
 * const char *clauses[4] = {"transform", "file", "function", 0};
 * int err = nomp_jit(&id, knl, clauses, 3, "a", sizeof(a[0]), NOMP_PTR, "b",
 *   sizeof(b[0]), NOMP_PTR, "N", sizeof(int), NOMP_INT);
 * @endcode
 *
 * @param[out] id Id of the generated kernel.
 * @param[in] csrc Kernel source in C.
 * @param[in] clauses Clauses to provide meta information about the kernel.
 * @param[in] nargs Number of arguments to the kernel.
 * @param[in] ... Three values for each argument: identifier, sizeof(argument)
 * and argument type.
 * @return int
 */
int nomp_jit(int *id, const char *csrc, const char **clauses, int nargs, ...) {
  if (*id >= 0)
    return 0;

  if (progs_n == progs_max) {
    progs_max += progs_max / 2 + 1;
    progs = nomp_realloc(progs, struct nomp_prog_t *, progs_max);
  }

  // Initialize the struct nomp_prog_t with the kernel input arguments.
  va_list args;
  va_start(args, nargs);
  struct nomp_prog_t *prg = init_args(progs_n, nargs, args);
  va_end(args);

  // Parse the clauses to find transformations file, function and other
  // annotations. Annotations are returned as a Python dictionary.
  struct nomp_meta_t m;
  nomp_check(parse_clauses(&m, prg, clauses));

  // Create loopy kernel from C source.
  PyObject *knl = NULL;
  nomp_check(nomp_py_c_to_loopy(&knl, csrc));

  // Handle annotate clauses if they exist.
  nomp_check(nomp_py_apply_annotations(&knl, nomp.py_annotate, m.dict,
                                       nomp.py_context));
  Py_XDECREF(m.dict);

  // Handle transform clauses.
  nomp_check(nomp_py_apply_transform(&knl, m.file, m.func, nomp.py_context));
  nomp_free(&m.file), nomp_free(&m.func);

  // Handle reductions if they exist.
  if (prg->redn_idx >= 0)
    nomp_check(nomp_py_realize_reduction(&knl, prg->args[prg->redn_idx].name,
                                         nomp.py_context));

  // Get OpenCL, CUDA, etc. source and name from the loopy kernel and build
  // the program.
  char *name, *src;
  nomp_check(nomp_py_get_knl_name_and_src(&name, &src, knl));
  nomp_check(nomp.knl_build(&nomp, prg, src, name));
  nomp_free(&src), nomp_free(&name);

  // Get grid size of the loopy kernel as pymbolic expressions. These grid
  // sizes will be evaluated each time the kernel is run.
  nomp_check(nomp_py_get_grid_size(prg, knl));
  Py_XDECREF(knl);

  *id = progs_n++;

  return 0;
}

/**
 * @ingroup nomp_user_api
 * @brief Runs the kernel generated by nomp_jit().
 *
 * @details Runs the kernel with a given kernel id. Kernel id is followed by the
 * arguments (i.e., pointers and pointer to scalar variables).
 *
 * <b>Example usage:</b>
 * @code{.c}
 * int N = 10;
 * double a[10], b[10];
 * for (unsigned i = 0; i < N; i++) {
 *   a[i] = i;
 *   b[i] = 10 -i
 * }
 *
 * static int id = -1;
 * const char *knl = "for (unsigned i = 0; i < N; i++) a[i] += b[i];"
 * const char *clauses[4] = {"transform", "file", "function", 0};
 * int err = nomp_jit(&id, knl, clauses, 3, "a", sizeof(a[0]), NOMP_PTR, "b",
 *   sizeof(b[0]), NOMP_PTR, "N", sizeof(int), NOMP_INT);
 * err = nomp_run(id, a, b, &N);
 * @endcode
 *
 * @param[in] id Id of the kernel to be run.
 * @param[in] ...  Arguments to the kernel.
 *
 * @return int
 */
int nomp_run(int id, ...) {
  if (id < 0) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Kernel id %d passed to nomp_run is not valid.", id);
  }

  struct nomp_prog_t *prg = progs[id];
  prg->eval_grid = 0;

  struct nomp_arg_t *args = prg->args;
  struct nomp_mem_t *m;
  long val;

  va_list vargs;
  va_start(vargs, id);
  for (unsigned i = 0; i < prg->nargs; i++) {
    args[i].ptr = va_arg(vargs, void *);
    switch (args[i].type) {
    case NOMP_INT:
      val = *((int *)args[i].ptr);
      prg->eval_grid |= nomp_symengine_update(prg->map, args[i].name, val);
      break;
    case NOMP_UINT:
      val = *((unsigned *)args[i].ptr);
      prg->eval_grid |= nomp_symengine_update(prg->map, args[i].name, val);
      break;
    case NOMP_PTR:
      m = mem_if_mapped(args[i].ptr);
      if (m == NULL) {
        if (prg->redn_idx == (int)i) {
          prg->redn_ptr = args[i].ptr;
          m = &nomp.scratch;
        } else {
          return nomp_log(NOMP_USER_MAP_PTR_IS_INVALID, NOMP_ERROR,
                          ERR_STR_USER_MAP_PTR_IS_INVALID, args[i].ptr);
        }
      }
      args[i].size = m->bsize, args[i].ptr = m->bptr;
      break;
    }
  }
  va_end(vargs);

  if (prg->eval_grid)
    nomp_check(nomp_symengine_eval_grid_size(prg));

  nomp_check(nomp.knl_run(&nomp, prg));
  if (prg->redn_idx >= 0)
    nomp_check(nomp_host_side_reduction(&nomp, prg, &nomp.scratch));

  return 0;
}

/**
 * @ingroup nomp_user_api
 * @brief Synchronize task execution on device.
 *
 * Implement a host-side barrier till the device finish executing all the
 * previous nomp kernels and/or memory copies.
 *
 * @return int
 */
int nomp_sync(void) { return nomp.sync(&nomp); }

/**
 * @ingroup nomp_user_api
 * @brief Finalizes libnomp runtime.
 *
 * @details Frees allocated runtime resources for libnomp. Returns a non-zero
 * value if an error occurs during the finalize process, otherwise returns 0.
 * Calling this method before nomp_init() will return an error. Calling this
 * method twice will also return an error.
 *
 * @return int
 */
int nomp_finalize(void) {
  // Free bookkeeping structures for the logger and profiler since these can be
  // released irrespective of whether libnomp is initialized or not.
  nomp_log_finalize();
  nomp_profile_finalize();

  if (!initialized)
    return NOMP_FINALIZE_FAILURE;

  Py_XDECREF(nomp.py_annotate), Py_XDECREF(nomp.py_context), Py_Finalize();

  for (unsigned i = 0; i < mems_n; i++) {
    if (mems[i]) {
      nomp_check(nomp.update(&nomp, mems[i], NOMP_FREE, mems[i]->idx0,
                             mems[i]->idx1, mems[i]->usize));
      nomp_free(&mems[i]);
    }
  }
  nomp_free(&mems), mems_n = mems_max = 0;
  nomp_check(deallocate_scratch_memory(&nomp));

  for (unsigned i = 0; i < progs_n; i++) {
    if (progs[i]) {
      nomp_check(nomp.knl_free(progs[i]));
      vecbasic_free(progs[i]->sym_global), vecbasic_free(progs[i]->sym_local);
      mapbasicbasic_free(progs[i]->map), nomp_free(&progs[i]->args);
    }
    nomp_free(&progs[i]);
  }
  nomp_free(&progs), progs_n = progs_max = 0;

  if ((initialized = nomp.finalize(&nomp)))
    return NOMP_FINALIZE_FAILURE;
  return 0;
}
