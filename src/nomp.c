#include "nomp-impl.h"
#include "nomp-reduction.h"

static struct nomp_backend_t nomp;
static int initialized = 0;

static inline int check_cmd_line_aux(unsigned i, unsigned argc,
                                     const char *argv[]) {
  if (i >= argc || argv[i] == NULL) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Missing argument value after: %s.", argv[i - 1]);
  }
  return 0;
}

static inline int check_cmd_line(struct nomp_backend_t *bnd, int argc,
                                 const char **argv) {
  if (argc <= 1 || argv == NULL)
    return 0;

  unsigned i = 0;
  while (i < argc) {
    if (!strncmp("--nomp", argv[i], 6)) {
      nomp_check(check_cmd_line_aux(i + 1, argc, argv));
      if (!strncmp("--nomp-backend", argv[i], NOMP_MAX_BUFSIZ)) {
        strncpy(bnd->backend, argv[i + 1], NOMP_MAX_BUFSIZ);
      } else if (!strncmp("--nomp-platform", argv[i], NOMP_MAX_BUFSIZ)) {
        bnd->platform_id = nomp_str_toui(argv[i + 1], NOMP_MAX_BUFSIZ);
      } else if (!strncmp("--nomp-device", argv[i], NOMP_MAX_BUFSIZ)) {
        bnd->device_id = nomp_str_toui(argv[i + 1], NOMP_MAX_BUFSIZ);
      } else if (!strncmp("--nomp-verbose", argv[i], NOMP_MAX_BUFSIZ)) {
        bnd->verbose = nomp_str_toui(argv[i + 1], NOMP_MAX_BUFSIZ);
      } else if (!strncmp("--nomp-profile", argv[i], NOMP_MAX_BUFSIZ)) {
        bnd->profile = nomp_str_toui(argv[i + 1], NOMP_MAX_BUFSIZ);
      } else if (!strncmp("--nomp-install-dir", argv[i], NOMP_MAX_BUFSIZ)) {
        strncpy(bnd->install_dir, argv[i + 1], PATH_MAX);
      } else if (!strncmp("--nomp-function", argv[i], NOMP_MAX_BUFSIZ)) {
        nomp_check(nomp_py_set_annotate_func(&bnd->py_annotate,
                                             (const char *)argv[i + 1]));
      }
      i++;
    }
    i++;
  }

  return 0;
}

static inline int check_env_vars(struct nomp_backend_t *bnd) {
  char *tmp = getenv("NOMP_PLATFORM");
  if (tmp)
    bnd->platform_id = nomp_str_toui(tmp, NOMP_MAX_BUFSIZ);

  if ((tmp = getenv("NOMP_DEVICE")))
    bnd->device_id = nomp_str_toui(tmp, NOMP_MAX_BUFSIZ);

  if ((tmp = getenv("NOMP_VERBOSE")))
    bnd->verbose = nomp_str_toui(tmp, NOMP_MAX_BUFSIZ);

  if ((tmp = getenv("NOMP_PROFILE")))
    bnd->profile = nomp_str_toui(tmp, NOMP_MAX_BUFSIZ);

  if ((tmp = getenv("NOMP_ANNOTATE_FUNCTION")))
    nomp_check(nomp_py_set_annotate_func(&bnd->py_annotate, tmp));

  if ((tmp = getenv("NOMP_BACKEND")))
    strncpy(bnd->backend, tmp, NOMP_MAX_BUFSIZ);

  if ((tmp = getenv("NOMP_INSTALL_DIR")))
    strncpy(bnd->install_dir, tmp, PATH_MAX);

  return 0;
}

static inline int init_configs(int argc, const char **argv,
                               struct nomp_backend_t *bnd) {
  // verbose, profile, device and platform id are all initialized to zero.
  // Everything else has to be set by user explicitly.
  bnd->verbose = bnd->profile = bnd->device_id = bnd->platform_id = 0;
  strcpy(bnd->backend, ""), strcpy(bnd->install_dir, "");

  nomp_check(check_cmd_line(bnd, argc, argv));
  nomp_check(check_env_vars(bnd));

#define check_if_initialized(COND, CMDARG, ENVVAR)                             \
  {                                                                            \
    if (COND) {                                                                \
      return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,                  \
                      #ENVVAR " is missing or invalid. Set it with " CMDARG    \
                              " command line argument or " ENVVAR              \
                              " environment variable.");                       \
    }                                                                          \
  }

  check_if_initialized(strlen(bnd->backend) == 0, "--nomp-backend",
                       "NOMP_BACKEND");
  check_if_initialized(strlen(bnd->install_dir) == 0, "--nomp-install-dir",
                       "NOMP_INSTALL_DIR");

#undef check_if_initialized

  // Append nomp python directory to sys.path.
  char abs_dir[PATH_MAX + 32];
  strncpy(abs_dir, bnd->install_dir, PATH_MAX);
  strncat(abs_dir, "/python", 32);
  nomp_check(nomp_py_append_to_sys_path(abs_dir));
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

static inline int init_backend(struct nomp_backend_t *bnd) {
  size_t n = strnlen(bnd->backend, NOMP_MAX_BUFSIZ);
  for (int i = 0; i < n; i++)
    bnd->backend[i] = tolower(bnd->backend[i]);

  bnd->py_context = PyDict_New();
  PyObject *obj = PyUnicode_FromString(bnd->backend);
  PyDict_SetItemString(bnd->py_context, "backend::name", obj);
  Py_XDECREF(obj);

  if (strncmp(bnd->backend, "opencl", NOMP_MAX_BUFSIZ) == 0) {
#if defined(OPENCL_ENABLED)
    nomp_check(opencl_init(&nomp, bnd->platform_id, bnd->device_id));
#endif
  } else if (strncmp(bnd->backend, "cuda", NOMP_MAX_BUFSIZ) == 0) {
#if defined(CUDA_ENABLED)
    nomp_check(cuda_init(&nomp, bnd->platform_id, bnd->device_id));
#endif
  } else if (strncmp(bnd->backend, "hip", NOMP_MAX_BUFSIZ) == 0) {
#if defined(HIP_ENABLED)
    nomp_check(hip_init(&nomp, bnd->platform_id, bnd->device_id));
#endif
  } else if (strncmp(bnd->backend, "sycl", NOMP_MAX_BUFSIZ) == 0) {
#if defined(SYCL_ENABLED)
    nomp_check(sycl_init(&nomp, bnd->platform_id, bnd->device_id));
#endif
  } else if (strncmp(bnd->backend, "ispc", NOMP_MAX_BUFSIZ) == 0) {
#if defined(ISPC_ENABLED)
    nomp_check(ispc_init(&nomp, bnd->platform_id, bnd->device_id));
#endif
  } else {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Invalid backend: %s.", bnd->backend);
  }

  return 0;
}

int nomp_init(int argc, const char **argv) {
  if (initialized) {
    return nomp_log(NOMP_INITIALIZE_FAILURE, NOMP_ERROR,
                    "libnomp is already initialized.");
  }

  if (!Py_IsInitialized()) {
    // May be we need the isolated configuration listed here:
    // https://docs.python.org/3/c-api/init_config.html#init-config
    // But for now, we do the simplest thing possible.
    Py_InitializeEx(0);
    // Append current working directory to sys.path.
    nomp_check(nomp_py_append_to_sys_path("."));
  }

  nomp_check(init_configs(argc, argv, &nomp));

  // Set profile level.
  nomp_check(nomp_profile_set_level(nomp.profile));

  // Set verbose level.
  nomp_check(nomp_log_set_verbose(nomp.verbose));

  // Maybe we shouldn't profile nomp_init(). But for now, we do.
  nomp_profile("nomp_init", 1, 0);

  // Initialize the backend.
  nomp_check(init_backend(&nomp));

  // Allocate scratch memory.
  nomp_check(allocate_scratch_memory(&nomp));

  initialized = 1;

  nomp_profile("nomp_init", 0, 0);

  return 0;
}

static struct nomp_mem_t **mems = NULL;
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

int nomp_update(void *ptr, size_t idx0, size_t idx1, size_t usize,
                nomp_map_direction_t op) {
  nomp_profile("nomp_update", 1, 1);
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

  nomp_profile("nomp_update", 0, 1);
  return 0;
}

static struct nomp_prog_t **progs = NULL;
static int progs_n = 0;
static int progs_max = 0;

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
    if (strncmp(clauses[i], "transform", NOMP_MAX_BUFSIZ) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "\"transform\" clause should be followed "
                        "by a file name and a "
                        "function name. At least one of them is "
                        "not provided.");
      }
      nomp_check(nomp_check_py_script_path((const char *)clauses[i + 1]));
      meta->file = strndup(clauses[i + 1], PATH_MAX);
      meta->func = strndup(clauses[i + 2], NOMP_MAX_BUFSIZ);
      i += 3;
    } else if (strncmp(clauses[i], "annotate", NOMP_MAX_BUFSIZ) == 0) {
      if (clauses[i + 1] == NULL || clauses[i + 2] == NULL) {
        return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "\"annotate\" clause should be followed by "
                        "a key value "
                        "pair. At least one of them is not "
                        "provided.");
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
        return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                        "\"reduce\" clause should be followed by a "
                        "variable name and an "
                        "operation. At least one of them is not "
                        "provided.");
      }
      for (unsigned j = 0; j < prg->nargs; j++) {
        if (strncmp(prg->args[j].name, clauses[i + 1], NOMP_MAX_BUFSIZ) == 0) {
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
    } else if (strncmp(clauses[i], "pin", NOMP_MAX_BUFSIZ) == 0) {
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
    strncpy(prg->args[i].name, va_arg(args, const char *), NOMP_MAX_BUFSIZ);
    prg->args[i].size = va_arg(args, size_t);
    prg->args[i].type = va_arg(args, int);
  }
  return prg;
}

int nomp_jit(int *id, const char *csrc, const char **clauses, int nargs, ...) {
  if (*id >= 0)
    return 0;

  nomp_profile("nomp_jit", 1, 1);
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
  nomp_check(nomp_py_c_to_loopy(&knl, csrc, nomp.backend));

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
  nomp_check(nomp_py_get_knl_name_and_src(&name, &src, knl, nomp.backend));
  nomp_check(nomp.knl_build(&nomp, prg, src, name));
  nomp_free(&src), nomp_free(&name);

  // Get grid size of the loopy kernel as pymbolic expressions. These grid
  // sizes will be evaluated each time the kernel is run.
  nomp_check(nomp_py_get_grid_size(prg, knl));
  Py_XDECREF(knl);

  *id = progs_n++;
  nomp_profile("nomp_jit", 0, 1);

  return 0;
}

int nomp_run(int id, ...) {
  if (id < 0) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Kernel id %d passed to nomp_run is not valid.", id);
  }

  nomp_profile("nomp_run setup time", 1, 1);
  struct nomp_prog_t *prg = progs[id];
  prg->eval_grid = 0;

  struct nomp_arg_t *args = prg->args;
  struct nomp_mem_t *m;
  int val, val_len;
  char str_val[64];

  va_list vargs;
  va_start(vargs, id);
  for (unsigned i = 0; i < prg->nargs; i++) {
    args[i].ptr = va_arg(vargs, void *);
    switch (args[i].type) {
    case NOMP_INT:
      val = *((int *)args[i].ptr);
      goto str_val;
      break;
    case NOMP_UINT:
      val = *((unsigned int *)args[i].ptr);
    str_val:
      snprintf(str_val, sizeof(str_val), "%d", val);
      prg->eval_grid |= nomp_symengine_update(prg->map, args[i].name, str_val);
      break;
    case NOMP_PTR:
      m = mem_if_mapped(args[i].ptr);
      if (m == NULL) {
        if (prg->redn_idx == i) {
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
  nomp_profile("nomp_run setup time", 0, 1);

  nomp_profile("nomp_run grid evaluation", 1, 1);
  if (prg->eval_grid)
    nomp_check(nomp_py_eval_grid_size(prg));
  nomp_profile("nomp_run grid evaluation", 0, 1);

  nomp_profile("nomp_run kernel runtime", 1, 1);
  nomp_check(nomp.knl_run(&nomp, prg));
  if (prg->redn_idx >= 0)
    nomp_check(nomp_host_side_reduction(&nomp, prg, &nomp.scratch));
  nomp_profile("nomp_run kernel runtime", 0, 1);

  return 0;
}

int nomp_sync() { return nomp.sync(&nomp); }

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
