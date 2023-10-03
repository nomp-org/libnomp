#include "nomp-impl.h"
#include <symengine/cwrapper.h>

static const char *module_loopy_api = "loopy_api";
static const char *module_reduction = "reduction";
static const char *c_to_loopy = "c_to_loopy";
static const char *get_knl_src = "get_knl_src";
static const char *get_knl_name = "get_knl_name";
static const char *realize_reduction = "realize_reduction";
static char backend[NOMP_MAX_BUFFER_SIZE + 1];

/**
 * @ingroup nomp_py_utils
 *
 * @brief Print the string representation of python object along with a debug
 * message to stderr.
 *
 * @param message Debug message to be printed before the object as a C-string.
 * @param obj Python object to be printed.
 * @return void
 */
void nomp_py_print(const char *const message, PyObject *const obj) {
  PyObject *py_repr = PyObject_Repr(obj);
  PyObject *py_str = PyUnicode_AsEncodedString(py_repr, "utf-8", "~E~");
  const char *str = PyBytes_AS_STRING(py_str);
  fprintf(stderr, "%s: %s\n", message, str);
  fflush(stderr);
  Py_XDECREF(py_repr), Py_XDECREF(py_str);
}

/**
 * @ingroup nomp_py_utils
 *
 * @brief Initialize the nomp python interface.
 *
 * @param[in] cfg Nomp configuration struct of type ::nomp_config_t.
 * @return int
 */
int nomp_py_init(const struct nomp_config_t *const cfg) {
  strncpy(backend, cfg->backend, NOMP_MAX_BUFFER_SIZE);

  if (!Py_IsInitialized()) {
    // May be we need the isolated configuration listed here:
    // https://docs.python.org/3/c-api/init_config.html#init-config
    // But for now, we do the simplest thing possible.
    Py_InitializeEx(0);
  }

  // Append current working directory to sys.path.
  nomp_check(nomp_py_append_to_sys_path("."));

  // Append nomp python directory to sys.path.
  char *py_dir = nomp_str_cat(2, PATH_MAX, cfg->install_dir, "/python");
  nomp_check(nomp_py_append_to_sys_path(py_dir));
  nomp_free(&py_dir);

  // Append nomp script directory to sys.path.
  nomp_check(nomp_py_append_to_sys_path(cfg->scripts_dir));

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Appends specified path to system path.
 *
 * @param[in] path Path to be appended to system path.
 * @return int
 */
int nomp_py_append_to_sys_path(const char *path) {
#define check_error(obj)                                                       \
  {                                                                            \
    if (!obj) {                                                                \
      return nomp_log(NOMP_PY_CALL_FAILURE, NOMP_ERROR,                        \
                      "Appending path \"%s\" to the sys.path failed.", path);  \
    }                                                                          \
  }

  PyObject *py_sys = PyImport_ImportModule("sys");
  check_error(py_sys);
  PyObject *py_path = PyObject_GetAttrString(py_sys, "path");
  check_error(py_path);
  PyObject *py_str = PyUnicode_FromString(path);
  check_error(py_str);
  check_error(!PyList_Append(py_path, py_str));

  Py_DECREF(py_path), Py_DECREF(py_str), Py_DECREF(py_sys);

#undef check_error

  return 0;
}

/**
 * @ingroup nomp_other_utils
 *
 * @brief Check if the given python module and function exist.
 *
 * Check if there is python function \p function exist in the python module
 * \p module.  Returns 0 if both module and function exist, otherwise returns
 * an error id which can beused to query the error id and string using
 * nomp_get_err_str() and nomp_get_err_id(). The \p module should be provided
 * without the ".py" extension.
 *
 * @param[in] module Python module name without the ".py" extension.
 * @param[in] function Python function name.
 * @return int
 */
int nomp_py_check_module(const char *module, const char *function) {
  if (module == NULL || function == NULL) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Module name and/or function name not provided.");
  }
  PyObject *py_str_module = PyUnicode_FromString(module);
  if (!py_str_module) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Can't convert string \"%s\" to a python string.", module);
  }
  PyObject *py_module = PyImport_Import(py_str_module);
  if (!py_module) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Python module \"%s\" not found.", module);
  }
  PyObject *py_function = PyObject_GetAttrString(py_module, function);
  if (!py_function) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Python function \"%s\" not found in module \"%s\".",
                    function, module);
  }

  Py_DECREF(py_function), Py_DECREF(py_module), Py_DECREF(py_str_module);
  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Realize reductions if present in the kernel.
 *
 * @param[in,out] kernel Loopy kernel object.
 * @param[in] var Name of the reduction variable as a C-string.
 * @param[in] py_context Python dictionary with context information.
 * @return int
 */
int nomp_py_realize_reduction(PyObject **kernel, const char *const var,
                              const PyObject *const py_context) {
#define check_error(obj)                                                       \
  {                                                                            \
    if (!obj) {                                                                \
      return nomp_log(NOMP_PY_CALL_FAILURE, NOMP_ERROR,                        \
                      "Call to realize_reduction() failed.");                  \
    }                                                                          \
  }

  PyObject *py_str_reduction = PyUnicode_FromString(module_reduction);
  check_error(py_str_reduction);
  PyObject *py_module = PyImport_Import(py_str_reduction);
  check_error(py_module);

  PyObject *py_realize_reduction =
      PyObject_GetAttrString(py_module, realize_reduction);
  check_error(py_realize_reduction);

  PyObject *py_str_var = PyUnicode_FromString(var);
  check_error(py_str_var);

  PyObject *py_result = PyObject_CallFunctionObjArgs(
      py_realize_reduction, *kernel, py_str_var, py_context, NULL);
  check_error(py_result);

  Py_DECREF(*kernel), *kernel = py_result;
  Py_DECREF(py_str_var), Py_DECREF(py_realize_reduction);
  Py_DECREF(py_module), Py_DECREF(py_str_reduction);

#undef check_error

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Creates loopy kernel from C source.
 *
 * @param[out] knl Loopy Kernel object.
 * @param[in] src C kernel source.
 * @return int
 */
int nomp_py_c_to_loopy(PyObject **knl, const char *src) {
#define check_error(obj)                                                       \
  {                                                                            \
    if (!obj) {                                                                \
      return nomp_log(NOMP_LOOPY_CONVERSION_FAILURE, NOMP_ERROR,               \
                      "C to Loopy conversion failed.\n");                      \
    }                                                                          \
  }

  PyObject *py_loopy_api = PyUnicode_FromString(module_loopy_api);
  check_error(py_loopy_api);

  PyObject *py_module = PyImport_Import(py_loopy_api);
  check_error(py_module);

  PyObject *py_c_to_loopy = PyObject_GetAttrString(py_module, c_to_loopy);
  check_error(py_c_to_loopy);

  PyObject *py_src = PyUnicode_FromString(src);
  check_error(py_src);

  PyObject *py_backend = PyUnicode_FromString(backend);
  check_error(py_backend);

  *knl = PyObject_CallFunctionObjArgs(py_c_to_loopy, py_src, py_backend, NULL);
  check_error(*knl);

  Py_XDECREF(py_src), Py_XDECREF(py_backend);
  Py_DECREF(py_c_to_loopy), Py_DECREF(py_module), Py_DECREF(py_loopy_api);

#undef check_error

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Get kernel name and generated source for the backend.
 *
 * @param[out] name Kernel name as a C-string.
 * @param[out] src Kernel source as a C-string.
 * @param[in] knl Loopy kernel object.
 * @return int
 */
int nomp_py_get_knl_name_and_src(char **name, char **src, const PyObject *knl) {
  int err = 1;
  PyObject *lpy_api = PyUnicode_FromString(module_loopy_api);
  if (lpy_api) {
    PyObject *module = PyImport_Import(lpy_api);
    if (module) {
      PyObject *knl_name = PyObject_GetAttrString(module, get_knl_name);
      if (knl_name) {
        PyObject *py_name = PyObject_CallFunctionObjArgs(knl_name, knl, NULL);
        if (py_name) {
          Py_ssize_t size;
          const char *name_ = PyUnicode_AsUTF8AndSize(py_name, &size);
          *name = strndup(name_, size);
          Py_DECREF(py_name), err = 0;
        }
        Py_DECREF(knl_name);
      }
      if (err) {
        return nomp_log(NOMP_LOOPY_KNL_NAME_NOT_FOUND, NOMP_ERROR,
                        "Unable to get loopy kernel name.");
      }

      err = 1;
      PyObject *knl_src = PyObject_GetAttrString(module, get_knl_src);
      if (knl_src) {
        PyObject *py_src = PyObject_CallFunctionObjArgs(knl_src, knl, NULL);
        if (py_src) {
          Py_ssize_t size;
          const char *src_ = PyUnicode_AsUTF8AndSize(py_src, &size);
          *src = strndup(src_, size);
          Py_DECREF(py_src), err = 0;
        }
        Py_DECREF(knl_src);
      }
      Py_DECREF(module);
    }
    Py_DECREF(lpy_api);
    if (err) {
      return nomp_log(
          NOMP_LOOPY_CODEGEN_FAILURE, NOMP_ERROR,
          "Backend code generation from loopy kernel \"%s\" failed.", *name);
    }
  }

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Apply transformations on a loopy kernel based on annotations.
 *
 * Apply the transformations to the loopy kernel \p kernel based on the
 * annotation function \p function and the key value pairs (annotations) passed
 * in \p annotations. \p kernel will be modified based on the transformations.
 *
 * @param[in,out] kernel Pointer to loopy kernel object.
 * @param[in] function Function which performs transformations based on
 * annotations.
 * @param[in] annotations Annotations (as a PyDict) to specify which
 * transformations to apply.
 * @param[in] context Context (as a PyDict) to pass around information such
 * as backend, device details, etc.
 * @return int
 */
int nomp_py_apply_annotations(PyObject **kernel, PyObject *const function,
                              const PyObject *const annotations,
                              const PyObject *const context) {
  if (!kernel || !*kernel || !function)
    return 0;

  if (!PyCallable_Check(function)) {
    return nomp_log(NOMP_PY_CALL_FAILURE, NOMP_ERROR,
                    "Annotation function is not callable.");
  }
  PyObject *py_temp = PyObject_CallFunctionObjArgs(function, *kernel,
                                                   annotations, context, NULL);
  Py_DECREF(*kernel), *kernel = py_temp;

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Apply kernel specific user transformations on a loopy kernel.
 *
 * Call the user transform function \p function in file \p file on the loopy
 * kernel \p kernel. \p kernel will be modified based on the transformations.
 * Python file (module) must reside on nomp scripts directory which is set
 * using `--nomp-scripts-dir` option or environment variable `NOMP_SCRIPTS_DIR`.
 * Function will return a non-zero value if there was an error. Non-zero return
 * value can be used to query the error code using nomp_get_err_no() and
 * error message using nomp_get_err_str().
 *
 * @param[in,out] kernel Pointer to loopy kernel object.
 * @param[in] file Name of the file containing transform function \p function.
 * @param[in] function Name of the transform function.
 * @param[in] context Context (as a Python dictionary) to pass around
 * information such as backend, device details, etc.
 * @return int
 */
int nomp_py_apply_transform(PyObject **kernel, const char *const file,
                            const char *function,
                            const PyObject *const context) {
  // If either kernel, file, or function are NULL, we don't have to do anything:
  if (kernel == NULL || *kernel == NULL || file == NULL || function == NULL)
    return 0;

#define check_error(obj)                                                       \
  {                                                                            \
    if (!obj) {                                                                \
      return nomp_log(                                                         \
          NOMP_PY_CALL_FAILURE, NOMP_ERROR,                                    \
          "Failed to call user transform function: \"%s\" in file: "           \
          "\"%s\".",                                                           \
          function, file);                                                     \
    }                                                                          \
  }

  PyObject *py_str_file = PyUnicode_FromString(file);
  check_error(py_str_file);

  PyObject *py_module = PyImport_Import(py_str_file);
  check_error(py_module);

  PyObject *py_function = PyObject_GetAttrString(py_module, function);
  check_error(py_function);
  check_error(PyCallable_Check(py_function));

  PyObject *py_temp =
      PyObject_CallFunctionObjArgs(py_function, *kernel, context, NULL);
  check_error(py_temp);
  Py_DECREF(*kernel), *kernel = py_temp;

  Py_DECREF(py_function);
  Py_DECREF(py_module);
  Py_DECREF(py_str_file);

  return 0;
}

static int symengine_vec_push(CVecBasic *vec, const char *str) {
  basic a;
  basic_new_stack(a);
  CWRAPPER_OUTPUT_TYPE err = basic_parse(a, str);
  if (err) {
    return nomp_log(NOMP_LOOPY_GRIDSIZE_FAILURE, NOMP_ERROR,
                    "Expression parsing with SymEngine failed with error %d.",
                    err);
  }
  vecbasic_push_back(vec, a);
  basic_free_stack(a);

  return 0;
}

static int py_get_grid_size_aux(PyObject *exp, CVecBasic *vec) {
  int err = 1;
  PyObject *mapper = PyImport_ImportModule("pymbolic.interop.symengine");
  if (mapper) {
    PyObject *module_name = PyUnicode_FromString("PymbolicToSymEngineMapper");
    PyObject *p2s_mapper = PyObject_CallMethodNoArgs(mapper, module_name);
    if (p2s_mapper) {
      PyObject *sym_exp = PyObject_CallFunctionObjArgs(p2s_mapper, exp, NULL);
      if (sym_exp) {
        PyObject *obj_rep = PyObject_Repr(sym_exp);
        const char *str = PyUnicode_AsUTF8(obj_rep);
        err = symengine_vec_push(vec, str);
        Py_XDECREF(obj_rep), Py_DECREF(sym_exp);
      }
      Py_DECREF(p2s_mapper);
    }
    Py_DECREF(mapper), Py_XDECREF(module_name);
  }
  if (err) {
    return nomp_log(NOMP_LOOPY_GRIDSIZE_FAILURE, NOMP_ERROR,
                    "Unable to evaluate grid sizes from loopy kernel.");
  }

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Get global and local grid sizes as `pymoblic` expressions.
 *
 * Grid sizes are stored in the program object itself.
 *
 * @param[in] prg Nomp program object.
 * @param[in] knl Python kernel object.
 * @return int
 */
int nomp_py_get_grid_size(struct nomp_prog_t *prg, PyObject *knl) {
  int err = 1;
  if (knl) {
    PyObject *callables = PyObject_GetAttrString(knl, "callables_table");
    if (callables) {
      PyObject *entry = PyObject_GetAttrString(knl, "default_entrypoint");
      if (entry) {
        PyObject *expr = PyObject_GetAttrString(
            entry, "get_grid_size_upper_bounds_as_exprs");
        if (expr) {
          PyObject *grid_size =
              PyObject_CallFunctionObjArgs(expr, callables, NULL);
          if (grid_size && PyTuple_Check(grid_size)) {
            PyObject *py_global = PyTuple_GetItem(grid_size, 0);
            PyObject *py_local = PyTuple_GetItem(grid_size, 1);
            prg->ndim =
                nomp_max(2, PyTuple_Size(py_global), PyTuple_Size(py_local));

            for (int i = 0; i < PyTuple_Size(py_global); i++)
              nomp_check(py_get_grid_size_aux(PyTuple_GetItem(py_global, i),
                                              prg->sym_global));

            for (int i = 0; i < PyTuple_Size(py_local); i++)
              nomp_check(py_get_grid_size_aux(PyTuple_GetItem(py_local, i),
                                              prg->sym_local));

            err = 0;
            Py_DECREF(grid_size);
          }
          Py_DECREF(expr);
        }
        Py_DECREF(entry);
      }
      Py_DECREF(callables);
    }
  }
  if (err) {
    return nomp_log(NOMP_LOOPY_GRIDSIZE_FAILURE, NOMP_ERROR,
                    "Unable to get grid sizes from loopy kernel.");
  }

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Set the annotate function based on the path to annotation script and
 * function.
 *
 * @param[out] annotate_func Pointer to the annotate function.
 * @param[in] path_ Path to the annotation script followed by function name
 * (path and function name must be separated by "::").
 * @return int
 */
int nomp_py_set_annotate_func(PyObject **annotate_func, const char *path_) {
  // Find file and function from path.
  char *path = strndup(path_, PATH_MAX + NOMP_MAX_BUFFER_SIZE);
  char *file = strtok(path, "::"), *func = strtok(NULL, "::");
  if (file == NULL || func == NULL) {
    nomp_free(&path);
    return 0;
  }

  // nomp_check(nomp_py_check_module(file));

  int err = 1;
  PyObject *pfile = PyUnicode_FromString(file);
  if (pfile) {
    PyObject *module = PyImport_Import(pfile);
    if (module) {
      PyObject *pfunc = PyObject_GetAttrString(module, func);
      if (pfunc && PyCallable_Check(pfunc))
        Py_XDECREF(*annotate_func), *annotate_func = pfunc, err = 0;
      Py_DECREF(module);
    }
    Py_DECREF(pfile);
  }
  if (err) {
    err = nomp_log(NOMP_PY_CALL_FAILURE, NOMP_ERROR,
                   "Failed to find annotate function \"%s\" in file \"%s\".",
                   func, file);
  }

  nomp_free(&path);
  return err;
}
