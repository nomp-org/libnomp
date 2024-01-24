#include "nomp-aux.h"
#include "nomp-impl.h"
#include "nomp-loopy.h"

static char backend[NOMP_MAX_BUFFER_SIZE + 1];
static PyObject *py_backend_str = NULL;
static PyObject *py_pymbolic_to_symengine_str = NULL;

#define check_error_(obj, err, msg)                                            \
  {                                                                            \
    if (!obj)                                                                  \
      return nomp_log(err, NOMP_ERROR, msg);                                   \
  }

#define check_py_str(obj)                                                      \
  check_error_(obj, NOMP_PY_CALL_FAILURE,                                      \
               "Converting C string to python string failed.")

#define check_py_call(err, msg) check_error_(err, NOMP_PY_CALL_FAILURE, msg)

/**
 * @ingroup nomp_py_utils
 *
 * @brief Initialize the nomp python interface.
 *
 * @param[in] cfg Nomp configuration struct of type ::nomp_config_t.
 * @return int
 */
int nomp_py_init(const nomp_config_t *const cfg) {
  strncpy(backend, cfg->backend, NOMP_MAX_BUFFER_SIZE);
  backend[NOMP_MAX_BUFFER_SIZE] = '\0';

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

  py_backend_str = PyUnicode_FromString(backend);
  check_py_str(py_backend_str);

  py_pymbolic_to_symengine_str =
      PyUnicode_FromString("PymbolicToSymEngineMapper");
  check_py_str(py_pymbolic_to_symengine_str);

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
  PyObject *py_sys = PyImport_ImportModule("sys");
  check_py_call(py_sys, "Importing sys module failed.");

  PyObject *py_sys_path = PyObject_GetAttrString(py_sys, "path");
  check_py_call(py_sys_path, "Importing sys.path failed.");

  PyObject *py_path_str = PyUnicode_FromString(path);
  check_py_str(py_path_str);

  char msg[NOMP_MAX_BUFFER_SIZE + 1];
  snprintf(msg, NOMP_MAX_BUFFER_SIZE,
           "Appending path \"%s\" to the sys.path failed.", path);
  check_py_call(!PyList_Append(py_sys_path, py_path_str), msg);

  Py_DECREF(py_sys_path), Py_DECREF(py_path_str), Py_DECREF(py_sys);

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
 * nomp_get_err_str() and nomp_get_err_no(). The \p module should be provided
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

  PyObject *py_module = PyImport_ImportModule(module);

  char msg[BUFSIZ];
  snprintf(msg, BUFSIZ, "Importing Python module \"%s\" failed.", module);
  check_py_call(py_module, msg);

  PyObject *py_function = PyObject_GetAttrString(py_module, function);

  snprintf(msg, BUFSIZ,
           "Importing Python function \"%s\" from module \"%s\" failed.",
           function, module);
  check_py_call(py_function, msg);

  Py_DECREF(py_function), Py_DECREF(py_module);

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Creates loopy kernel from C source.
 *
 * @param[out] kernel Loopy Kernel object.
 * @param[in] src C kernel source.
 * @return int
 */
int nomp_py_c_to_loopy(PyObject **kernel, const char *src) {
  PyObject *py_src_str = PyUnicode_FromString(src);
  check_py_str(py_src_str);

  PyObject *py_loopy_api = PyImport_ImportModule("loopy_api");
  check_py_call(py_loopy_api, "Importing loopy_api module failed.");

  PyObject *py_c_to_loopy = PyObject_GetAttrString(py_loopy_api, "c_to_loopy");
  check_py_call(py_c_to_loopy,
                "Importing c_to_loopy function from loop_api module failed.");

  *kernel = PyObject_CallFunctionObjArgs(py_c_to_loopy, py_src_str,
                                         py_backend_str, NULL);
  check_error_(*kernel, NOMP_LOOPY_CONVERSION_FAILURE,
               "Converting C source to loopy kernel failed.");

  Py_DECREF(py_loopy_api), Py_DECREF(py_c_to_loopy), Py_DECREF(py_src_str);

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Realize reductions if one is present in the kernel.
 *
 * @param[in,out] kernel Loopy kernel object.
 * @param[in] variable Name of the reduction variable as a C-string.
 * @param[in] py_context Python dictionary with context information.
 * @return int
 */
int nomp_py_realize_reduction(PyObject **kernel, const char *const variable,
                              const PyObject *const py_context) {
  PyObject *py_module = PyImport_ImportModule("reduction");
  check_py_call(py_module, "Importing reduction module failed.");

  PyObject *py_realize_reduction =
      PyObject_GetAttrString(py_module, "realize_reduction");
  check_py_call(py_realize_reduction, "Importing realize_reduction function "
                                      "from reduction module failed.");

  PyObject *py_variable_str = PyUnicode_FromString(variable);
  check_py_str(py_variable_str);

  PyObject *py_result = PyObject_CallFunctionObjArgs(
      py_realize_reduction, *kernel, py_variable_str, py_context, NULL);
  check_py_call(py_result, "Calling realize_reduction() function failed.");

  Py_DECREF(*kernel), *kernel = py_result;
  Py_DECREF(py_variable_str), Py_DECREF(py_realize_reduction);
  Py_DECREF(py_module);

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
int nomp_py_transform(PyObject **kernel, const char *const file,
                      const char *function, const PyObject *const context) {
  // If either file, or function are NULL, we don't have to do anything:
  if (file == NULL || function == NULL)
    return 0;

  PyObject *py_module = PyImport_ImportModule(file);

  char msg[BUFSIZ];
  snprintf(msg, BUFSIZ, "Importing Python module: \"%s\" failed.", file);
  check_py_call(py_module, msg);

  PyObject *py_function = PyObject_GetAttrString(py_module, function);

  snprintf(msg, BUFSIZ,
           "Importing Python function \"%s\" from  module \"%s\" failed.",
           function, file);
  check_py_call(py_function, msg);

  snprintf(msg, BUFSIZ,
           "Python function \"%s\" from  module \"%s\" is not callable.",
           function, file);
  check_py_call(PyCallable_Check(py_function), msg);

  PyObject *py_transformed_kernel =
      PyObject_CallFunctionObjArgs(py_function, *kernel, context, NULL);

  snprintf(msg, BUFSIZ,
           "Calling Python function \"%s\" from module \"%s\" failed.",
           function, file);
  check_py_call(py_transformed_kernel, msg);

  Py_DECREF(*kernel), *kernel = py_transformed_kernel;

  Py_DECREF(py_function), Py_DECREF(py_module);

#undef check_error

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Get kernel name and generated source for the backend.
 *
 * @param[out] name Kernel name as a C-string.
 * @param[out] src Kernel source as a C-string.
 * @param[in] kernel Loopy kernel object.
 * @return int
 */
int nomp_py_get_knl_name_and_src(char **name, char **src,
                                 const PyObject *kernel) {
  PyObject *py_loopy_api = PyImport_ImportModule("loopy_api");
  check_py_call(py_loopy_api, "Importing module loopy_api failed.");

  PyObject *py_get_knl_name =
      PyObject_GetAttrString(py_loopy_api, "get_knl_name");
  check_py_call(py_get_knl_name,
                "Importing function loop_api.get_knl_name failed.");

  PyObject *py_name =
      PyObject_CallFunctionObjArgs(py_get_knl_name, kernel, NULL);
  check_error_(py_name, NOMP_LOOPY_KNL_NAME_NOT_FOUND,
               "Unable to get loopy kernel name.");

  Py_ssize_t size;
  const char *const name_ = PyUnicode_AsUTF8AndSize(py_name, &size);
  *name = strndup(name_, size);

  Py_DECREF(py_name), Py_DECREF(py_get_knl_name);

  PyObject *py_kernel_src = PyObject_GetAttrString(py_loopy_api, "get_knl_src");
  check_py_call(py_kernel_src,
                "Importing function loop_api.get_knl_src failed.");

  PyObject *py_src = PyObject_CallFunctionObjArgs(py_kernel_src, kernel, NULL);

  char msg[BUFSIZ];
  snprintf(msg, BUFSIZ,
           "Backend code generation from loopy kernel \"%s\" failed.", *name);
  check_error_(py_src, NOMP_LOOPY_CODEGEN_FAILURE, msg);

  const char *const src_ = PyUnicode_AsUTF8AndSize(py_src, &size);
  *src = strndup(src_, size);

  Py_DECREF(py_src), Py_DECREF(py_kernel_src);

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Set the annotate function based on the path to annotation script and
 * function.
 *
 * @param[out] annotate_func Pointer to the annotate function.
 * @param[in] file Name of the annotation script.
 * @return int
 */
int nomp_py_set_annotate_func(PyObject **annotate_func, const char *file) {
  // If file is NULL or empty, we don't have to do anything:
  if (file == NULL || strlen(file) == 0)
    return 0;

  PyObject *py_module = PyImport_ImportModule(file);
  check_py_call(py_module, "Importing python module failed.");

  PyObject *py_func = PyObject_GetAttrString(py_module, "annotate");
  char msg[BUFSIZ];
  snprintf(msg, BUFSIZ, "Failed to find annotate function in file \"%s\".",
           file);
  check_py_call(py_func, msg);

  check_py_call(PyCallable_Check(py_func),
                "Annotate function is not callable.");
  Py_XDECREF(*annotate_func), *annotate_func = py_func;

  Py_DECREF(py_module);

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
int nomp_py_annotate(PyObject **kernel, PyObject *const function,
                     const PyObject *const annotations,
                     const PyObject *const context) {
  // if kernel or function is NULL, we don't have to do anything:
  if (!kernel || !*kernel || !function)
    return 0;

  check_py_call(PyCallable_Check(function), "Annotation function is not "
                                            "callable.");
  PyObject *py_annotated_kernel = PyObject_CallFunctionObjArgs(
      function, *kernel, annotations, context, NULL);
  check_py_call(py_annotated_kernel, "Annotating loopy kernel failed.");
  Py_DECREF(*kernel), *kernel = py_annotated_kernel;

  return 0;
}

static int symengine_vec_push(CVecBasic *vec, const char *str) {
  basic a;
  basic_new_stack(a);

  CWRAPPER_OUTPUT_TYPE err = basic_parse(a, str);
  char msg[BUFSIZ];
  snprintf(msg, BUFSIZ,
           "Expression parsing with SymEngine failed with error %d.", err);
  check_error_(!err, NOMP_LOOPY_GRIDSIZE_FAILURE, msg);

  vecbasic_push_back(vec, a);
  basic_free_stack(a);

  return 0;
}

static int py_get_grid_size_aux(PyObject *exp, CVecBasic *vec) {
  PyObject *py_pymbolic = PyImport_ImportModule("pymbolic.interop.symengine");
  check_py_call(py_pymbolic,
                "Importing module pymbolic.interop.symengine failed.");

  PyObject *py_pymolic_to_symengine_mapper =
      PyObject_CallMethodNoArgs(py_pymbolic, py_pymbolic_to_symengine_str);
  check_py_call(py_pymolic_to_symengine_mapper,
                "Calling PymbolicToSymEngineMapper() failed.");

  PyObject *py_symengine_expr =
      PyObject_CallFunctionObjArgs(py_pymolic_to_symengine_mapper, exp, NULL);
  check_py_call(py_symengine_expr,
                "Converting pymbolic expression to SymEngine failed.");

  PyObject *py_expr_str = PyObject_Repr(py_symengine_expr);
  check_py_call(py_expr_str,
                "Converting SymEngine expression to string failed.");

  const char *str = PyUnicode_AsUTF8(py_expr_str);
  if (symengine_vec_push(vec, str)) {
    return nomp_log(NOMP_LOOPY_GRIDSIZE_FAILURE, NOMP_ERROR,
                    "Unable to evaluate grid sizes from loopy kernel.");
  }

  Py_DECREF(py_expr_str), Py_DECREF(py_symengine_expr);
  Py_DECREF(py_pymolic_to_symengine_mapper), Py_DECREF(py_pymbolic);

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Get global and local grid sizes as `pymoblic` expressions.
 *
 * Grid sizes are stored in the program object itself.
 *
 * @param[in] prg Nomp program object.
 * @param[in] kernel Python kernel object.
 * @return int
 */
int nomp_py_get_grid_size(nomp_prog_t *prg, PyObject *kernel) {
  check_py_call(kernel, "Loopy kernel object is NULL.");

  PyObject *py_callables = PyObject_GetAttrString(kernel, "callables_table");
  check_py_call(py_callables, "Loopy kernel's callables_table is NULL.");

  PyObject *py_entry = PyObject_GetAttrString(kernel, "default_entrypoint");
  check_py_call(py_entry, "Loopy kernel's default_entrypoint is NULL.");

  PyObject *py_grid_size_expr =
      PyObject_GetAttrString(py_entry, "get_grid_size_upper_bounds_as_exprs");
  check_py_call(
      py_grid_size_expr,
      "Loopy kernel's get_grid_size_upper_bounds_as_exprs() is NULL.");

  PyObject *py_grid_size =
      PyObject_CallFunctionObjArgs(py_grid_size_expr, py_callables, NULL);
  check_py_call(py_grid_size,
                "Calling get_grid_size_upper_bounds_as_exprs() failed.");
  check_py_call(PyTuple_Check(py_grid_size), "Grid size is not a tuple.");

  PyObject *py_global = PyTuple_GetItem(py_grid_size, 0);
  check_py_call(PyTuple_Check(py_global), "Global grid size is not a tuple.");

  PyObject *py_local = PyTuple_GetItem(py_grid_size, 1);
  check_py_call(PyTuple_Check(py_local), "Local grid size is not a tuple.");

  prg->ndim = nomp_max(2, PyTuple_Size(py_global), PyTuple_Size(py_local));

  for (int i = 0; i < PyTuple_Size(py_global); i++)
    nomp_check(
        py_get_grid_size_aux(PyTuple_GetItem(py_global, i), prg->sym_global));

  for (int i = 0; i < PyTuple_Size(py_local); i++)
    nomp_check(
        py_get_grid_size_aux(PyTuple_GetItem(py_local, i), prg->sym_local));

  Py_DECREF(py_grid_size), Py_DECREF(py_grid_size_expr), Py_DECREF(py_entry);
  Py_DECREF(py_callables);

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Fix the arguments which were marked as `jit` in the kernel.
 *
 * @param[in,out] kernel Python kernel object.
 * @param[in] py_dict Dictionary containing jit argument names and values.
 * @return int
 */
int nomp_py_fix_parameters(PyObject **kernel, const PyObject *py_dict) {
  PyObject *py_loopy_api = PyImport_ImportModule("loopy_api");

  PyObject *py_fix_parameters =
      PyObject_GetAttrString(py_loopy_api, "fix_parameters");
  check_py_call(py_fix_parameters,
                "Importing loopy_api.fix_parameters() failed.");

  check_py_call(PyCallable_Check(py_fix_parameters),
                "loopy_api.fix_parameters() is not callable.");

  PyObject *py_fixed_kernel =
      PyObject_CallFunctionObjArgs(py_fix_parameters, *kernel, py_dict, NULL);
  check_py_call(py_fixed_kernel, "Calling loopy.fix_parameters() failed.");

  Py_DECREF(*kernel), *kernel = py_fixed_kernel;

  Py_DECREF(py_fix_parameters);

  return 0;
}

/**
 * @ingroup nomp_py_utils
 *
 * @brief Get the string representation of a Python object.
 *
 * @param obj Python object.
 * @return char*
 */
char *nomp_py_get_str(PyObject *const obj) {
  PyObject *py_str = PyObject_Str(obj);
  Py_ssize_t size;
  const char *str_ = PyUnicode_AsUTF8AndSize(py_str, &size);
  char *str = nomp_calloc(char, size + 1);
  strncpy(str, str_, size);
  Py_XDECREF(py_str);
  return str;
}

/**
 * @ingroup nomp_py_utils
 * @brief Finalize the nomp python interface.
 *
 * @return int
 */
int nomp_py_finalize(void) {
  Py_XDECREF(py_pymbolic_to_symengine_str), py_pymbolic_to_symengine_str = NULL;
  Py_XDECREF(py_backend_str), py_backend_str = NULL;
  Py_Finalize();
  return 0;
}

#undef check_py_call
#undef check_py_str
#undef check_error_
