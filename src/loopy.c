#include "nomp-impl.h"
#include <symengine/cwrapper.h>

static const char *module_loopy_api = "loopy_api";
static const char *module_reduction = "reduction";

static const char *c_to_loopy = "c_to_loopy";
static const char *get_knl_src = "get_knl_src";
static const char *get_knl_name = "get_knl_name";
static const char *realize_reduction = "realize_reduction";

/**
 * @ingroup nomp_py_utils
 * @brief Get the string representation of python object.
 *
 * @param msg Debug message before printing the object.
 * @param obj Python object.
 * @return void
 */
void nomp_py_print(const char *msg, PyObject *obj) {
  PyObject *repr = PyObject_Repr(obj);
  PyObject *py_str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
  const char *str = PyBytes_AS_STRING(py_str);
  printf("%s: %s\n", msg, str);
  Py_XDECREF(repr), Py_XDECREF(py_str);
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
  if (!py_sys)
    goto err;
  PyObject *py_path = PyObject_GetAttrString(py_sys, "path");
  if (!py_path)
    goto err;
  PyObject *py_str = PyUnicode_FromString(path);
  if (PyList_Append(py_path, py_str))
    goto err;
  Py_DECREF(py_path), Py_XDECREF(py_str), Py_DECREF(py_sys);

  return 0;
err:
  return nomp_log(NOMP_PY_CALL_FAILURE, NOMP_ERROR,
                  "Appending path \"%s\" to the sys.path failed.", path);
}

/**
 * @ingroup nomp_py_utils
 * @brief Realize reductions if present in the kernel.
 *
 * @param[in,out] knl Loopy kernel object.
 * @param[in] var Name of the reduction variable.
 * @param[in] context Python dictionary with context information.
 * @return int
 */
int nomp_py_realize_reduction(PyObject **knl, const char *var,
                              const PyObject *context) {
  int err = 1;
  PyObject *reduction = PyUnicode_FromString(module_reduction);
  if (reduction) {
    PyObject *module = PyImport_Import(reduction);
    if (module) {
      PyObject *rr = PyObject_GetAttrString(module, realize_reduction);
      if (rr) {
        PyObject *pvar = PyUnicode_FromString(var);
        PyObject *result =
            PyObject_CallFunctionObjArgs(rr, *knl, pvar, context, NULL);
        if (result)
          Py_DECREF(*knl), *knl = result, err = 0;
        Py_XDECREF(pvar), Py_DECREF(rr);
      }
      Py_DECREF(module);
    }
    Py_DECREF(reduction);
  }
  if (err) {
    return nomp_log(NOMP_PY_CALL_FAILURE, NOMP_ERROR,
                    "Call to realize_reduction() failed.");
  }

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Creates loopy kernel from C source.
 *
 * @param[out] knl Loopy kernel object.
 * @param[in] src C kernel source.
 * @param[in] backend Backend name.
 * @return int
 */
int nomp_py_c_to_loopy(PyObject **knl, const char *src, const char *backend) {
  int err = 1;
  PyObject *lpy_api = PyUnicode_FromString(module_loopy_api);
  if (lpy_api) {
    PyObject *module = PyImport_Import(lpy_api);
    if (module) {
      PyObject *c_to_lpy = PyObject_GetAttrString(module, c_to_loopy);
      if (c_to_lpy) {
        PyObject *psrc = PyUnicode_FromString(src);
        PyObject *pbackend = PyUnicode_FromString(backend);
        *knl = PyObject_CallFunctionObjArgs(c_to_lpy, psrc, pbackend, NULL);
        err = (*knl == NULL);
        Py_XDECREF(psrc), Py_XDECREF(pbackend), Py_DECREF(c_to_lpy);
      }
      Py_DECREF(module);
    }
    Py_DECREF(lpy_api);
  }
  if (err) {
    return nomp_log(NOMP_LOOPY_CONVERSION_FAILURE, NOMP_ERROR,
                    "C to Loopy conversion failed.\n");
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

  nomp_check(nomp_check_py_script_path(file));

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

/**
 * @ingroup nomp_py_utils
 * @brief Apply transformations on a loopy kernel based on annotations.
 *
 * Apply the transformations to the loopy kernel \p knl based on the annotation
 * function \p func and the key value pairs (annotations) passed in \p annts.
 * \p knl will be modified based on the transformations.
 *
 * @param[in,out] knl Pointer to loopy kernel object.
 * @param[in] func Function which performs transformations based on annotations.
 * @param[in] annts Annotations (as a PyDict) to specify which transformations
 * to apply.
 * @param[in] context Context (as a PyDict) to pass around information such
 * as backend, device details, etc.
 * @return int
 */
int nomp_py_apply_annotations(PyObject **knl, PyObject *func,
                              const PyObject *annts, const PyObject *context) {
  if (knl && *knl && func) {
    if (PyCallable_Check(func)) {
      PyObject *tknl =
          PyObject_CallFunctionObjArgs(func, *knl, annts, context, NULL);
      if (tknl)
        Py_DECREF(*knl), *knl = tknl;
    }
  }

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Apply kernel specific user transformations on a loopy kernel.
 *
 * Call the user transform function \p func in file \p file on the loopy
 * kernel \p knl. \p knl will be modified based on the transformations.
 * Function will return a non-zero value if there was an error after
 * registering a log.
 *
 * @param[in,out] knl Pointer to loopy kernel object.
 * @param[in] file Path to the file containing transform function \p func.
 * @param[in] func Transform function.
 * @param[in] context Context (as a PyDict) to pass around information such
 * as backend, device details, etc.
 * @return int
 */
int nomp_py_apply_transform(PyObject **knl, const char *file, const char *func,
                            const PyObject *context) {
  // If either file, or func are NULL, we don't have to do anything:
  if (file == NULL || func == NULL)
    return 0;

  int err = 1;
  PyObject *pfile = PyUnicode_FromString(file);
  if (knl && *knl && pfile) {
    PyObject *module = PyImport_Import(pfile);
    if (module) {
      PyObject *pfunc = PyObject_GetAttrString(module, func);
      if (pfunc && PyCallable_Check(pfunc)) {
        PyObject *tknl =
            PyObject_CallFunctionObjArgs(pfunc, *knl, context, NULL);
        if (tknl)
          Py_DECREF(*knl), *knl = tknl, err = 0;
        Py_DECREF(pfunc);
      }
      Py_DECREF(module);
    }
    Py_DECREF(pfile);
  }
  if (err) {
    return nomp_log(
        NOMP_PY_CALL_FAILURE, NOMP_ERROR,
        "Failed to call user transform function: \"%s\" in file: \"%s\".", func,
        file);
  }

  return 0;
}

/**
 * @ingroup nomp_py_utils
 * @brief Get kernel name and generated source for the backend.
 *
 * @param[out] name Kernel name as a C-string.
 * @param[out] src Kernel source as a C-string.
 * @param[in] knl Loopy kernel object.
 * @param[in] backend Backend name.
 * @return int
 */
int nomp_py_get_knl_name_and_src(char **name, char **src, const PyObject *knl,
                                 const char *backend) {
  int err = 1;
  PyObject *lpy_api = PyUnicode_FromString(module_loopy_api);
  if (lpy_api) {
    PyObject *module = PyImport_Import(lpy_api);
    if (module) {
      PyObject *knl_name = PyObject_GetAttrString(module, get_knl_name);
      if (knl_name) {
        PyObject *py_backend = PyUnicode_FromString(backend);
        PyObject *py_name =
            PyObject_CallFunctionObjArgs(knl_name, knl, py_backend, NULL);
        if (py_name) {
          Py_ssize_t size;
          const char *name_ = PyUnicode_AsUTF8AndSize(py_name, &size);
          *name = strndup(name_, size);
          Py_DECREF(py_name), err = 0;
        }
        Py_XDECREF(py_backend), Py_DECREF(knl_name);
      }
      if (err) {
        return nomp_log(NOMP_LOOPY_KNL_NAME_NOT_FOUND, NOMP_ERROR,
                        "Unable to get loopy kernel name.");
      }

      err = 1;
      PyObject *knl_src = PyObject_GetAttrString(module, get_knl_src);
      if (knl_src) {
        PyObject *py_backend = PyUnicode_FromString(backend);
        PyObject *py_src =
            PyObject_CallFunctionObjArgs(knl_src, knl, py_backend, NULL);
        if (py_src) {
          Py_ssize_t size;
          const char *src_ = PyUnicode_AsUTF8AndSize(py_src, &size);
          *src = strndup(src_, size);
          Py_DECREF(py_src), err = 0;
        }
        Py_XDECREF(py_backend), Py_DECREF(knl_src);
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
