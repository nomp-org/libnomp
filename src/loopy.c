#include "nomp-impl.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

static const char *lpy_api = "loopy_api";
static const char *c_to_lpy = "c_to_loopy";
static const char *redn = "reduction";
static const char *rlze_redn = "realize_reduction";

void py_print(const char *msg, PyObject *obj) {
  PyObject *repr = PyObject_Repr(obj);
  PyObject *py_str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
  const char *str = PyBytes_AS_STRING(py_str);
  printf("%s: %s\n", msg, str);
  Py_XDECREF(repr), Py_XDECREF(py_str);
}

int py_append_to_sys_path(const char *path) {
  PyObject *sys = PyImport_ImportModule("sys");
  if (sys) {
    PyObject *ppath = PyObject_GetAttrString(sys, "path");
    Py_DECREF(sys);
    if (ppath) {
      PyObject *pstr = PyUnicode_FromString(path);
      int err = PyList_Append(ppath, pstr);
      Py_DECREF(ppath), Py_XDECREF(pstr);
      // TODO: Add the exception to the error message as well
      if (err)
        return set_log(NOMP_PY_CALL_FAILED, NOMP_ERROR,
                       "Appending path to the sys.path failed.");
    }
  }

  return 0;
}

int py_c_to_loopy(PyObject **knl, const char *src, const char *backend,
                  const char *redn_var) {
  check_null_input(src);
  check_null_input(backend);

  int err = 1;
  PyObject *string = PyUnicode_FromString(lpy_api);
  if (string) {
    PyObject *module = PyImport_Import(string);
    if (module) {
      PyObject *function = PyObject_GetAttrString(module, c_to_lpy);
      if (function) {
        PyObject *psrc = PyUnicode_FromString(src);
        PyObject *pbnd = PyUnicode_FromString(backend);
        PyObject *pvar = PyUnicode_FromString(redn_var);
        *knl = PyObject_CallFunctionObjArgs(function, psrc, pbnd, pvar, NULL);
        err = (*knl == NULL);
        Py_XDECREF(psrc), Py_XDECREF(pbnd), Py_XDECREF(pvar);
        Py_DECREF(function);
      }
      Py_DECREF(module);
    }
    Py_DECREF(string);
  }
  if (err) {
    return set_log(NOMP_LOOPY_CONVERSION_ERROR, NOMP_ERROR,
                   ERR_STR_LOOPY_CONVERSION_ERROR);
  }

  return 0;
}

int py_user_annotate(PyObject **knl, PyObject *annts, const char *file,
                     const char *func) {
  check_null_input(*knl);

  // If either file, or func are NULL, we don't have to do anything:
  if (file == NULL || func == NULL)
    return 0;

  int err = 1;
  PyObject *string = PyUnicode_FromString(file), *tk = NULL;
  if (string) {
    PyObject *module = PyImport_Import(string);
    Py_DECREF(string);
    if (module) {
      PyObject *function = PyObject_GetAttrString(module, func);
      Py_DECREF(module);
      if (function && PyCallable_Check(function)) {
        tk = PyObject_CallFunctionObjArgs(function, *knl, annts, NULL);
        Py_DECREF(function);
        err = (tk == NULL);
      }
    }
  }
  if (err) {
    return set_log(NOMP_PY_CALL_FAILED, NOMP_ERROR,
                   "Calling user annotate function: %s failed.", func);
  }

  Py_DECREF(*knl), *knl = tk;
  return 0;
}

int py_handle_reduction(PyObject **knl, const char *backend, const char *var) {
  check_null_input(*knl);

  int err = 1;
  PyObject *string = PyUnicode_FromString(redn), *tk = NULL;
  if (string) {
    PyObject *module = PyImport_Import(string);
    Py_DECREF(string);
    if (module) {
      PyObject *function = PyObject_GetAttrString(module, rlze_redn);
      Py_DECREF(module);
      if (function && PyCallable_Check(function)) {
        PyObject *pbackend = PyUnicode_FromString(backend);
        PyObject *pvar = PyUnicode_FromString(var);
        tk = PyObject_CallFunctionObjArgs(function, *knl, pbackend, pvar, NULL);
        err = (tk == NULL);
        Py_XDECREF(pbackend), Py_XDECREF(pvar), Py_DECREF(function);
      }
    }
  }
  if (err) {
    return set_log(NOMP_PY_CALL_FAILED, NOMP_ERROR,
                   "Calling realize_reduction failed.");
  }

  Py_DECREF(*knl), *knl = tk;
  return 0;
}

int py_user_transform(PyObject **knl, const char *file, const char *func) {
  check_null_input(*knl);

  // If either file, or func are NULL, we don't have to do anything:
  if (file == NULL || func == NULL)
    return 0;

  int err = 1;
  PyObject *string = PyUnicode_FromString(file), *tk = NULL;
  if (string) {
    PyObject *module = PyImport_Import(string);
    Py_DECREF(string);
    if (module) {
      PyObject *function = PyObject_GetAttrString(module, func);
      Py_DECREF(module);
      if (function && PyCallable_Check(function)) {
        tk = PyObject_CallFunctionObjArgs(function, *knl, NULL);
        Py_DECREF(function);
        err = (tk == NULL);
      }
    }
  }
  if (err) {
    return set_log(NOMP_PY_CALL_FAILED, NOMP_ERROR,
                   "Calling user transform function: \"%s\" failed.", func);
  }

  Py_DECREF(*knl), *knl = tk;
  return 0;
}

int py_get_knl_name(char **name, PyObject *knl) {
  int err = 1;
  Py_ssize_t size;
  PyObject *epts = PyObject_GetAttrString(knl, "entrypoints");
  if (epts) {
    Py_ssize_t len = PySet_Size(epts);
    // FIXME: This doesn't require iterator API
    // Iterator C API: https://docs.python.org/3/c-api/iter.html
    PyObject *iter = PyObject_GetIter(epts);
    if (iter) {
      PyObject *entry = PyIter_Next(iter);
      PyObject *py_name = PyObject_Str(entry);
      if (py_name) {
        const char *name_ = PyUnicode_AsUTF8AndSize(py_name, &size);
        *name = strndup(name_, size + 1);
        Py_DECREF(py_name), err = 0;
      }
      Py_XDECREF(entry), Py_DECREF(iter);
    }
    Py_DECREF(epts);
  }
  if (err) {
    return set_log(NOMP_LOOPY_KNL_NAME_NOT_FOUND, NOMP_ERROR,
                   "Failed to find loopy kernel name for kernel = %p.\n", knl);
  }

  return 0;
}

int py_get_knl_src(char **src, PyObject *knl) {
  int err = 1;
  Py_ssize_t size;
  PyObject *lpy = PyImport_ImportModule("loopy");
  if (lpy) {
    PyObject *gen_v2 = PyObject_GetAttrString(lpy, "generate_code_v2");
    if (gen_v2) {
      PyObject *code = PyObject_CallFunctionObjArgs(gen_v2, knl, NULL);
      if (code) {
        PyObject *py_device = PyObject_GetAttrString(code, "device_code");
        if (py_device) {
          PyObject *py_src = PyObject_CallFunctionObjArgs(py_device, NULL);
          if (py_src) {
            const char *src_ = PyUnicode_AsUTF8AndSize(py_src, &size);
            *src = strndup(src_, size + 1);
            Py_DECREF(py_src), err = 0;
          }
          Py_DECREF(py_device);
        }
        Py_DECREF(code);
      }
      Py_DECREF(gen_v2);
    }
    Py_DECREF(lpy);
  }
  if (err) {
    return set_log(NOMP_LOOPY_CODEGEN_FAILED, NOMP_ERROR,
                   "Failed to generate code from loopy kernel %p.", knl);
  }

  return 0;
}

int py_get_grid_size(struct prog *prg, PyObject *knl) {
  int err = 1;

  if (knl) {
    PyObject *callables = PyObject_GetAttrString(knl, "callables_table");
    if (callables) {
      // knl.default_entrypoint.get_grid_size_upper_bounds_as_exprs
      PyObject *entry = PyObject_GetAttrString(knl, "default_entrypoint");
      if (entry) {
        PyObject *expr = PyObject_GetAttrString(
            entry, "get_grid_size_upper_bounds_as_exprs");
        Py_DECREF(entry);
        if (expr) {
          PyObject *grid_size =
              PyObject_CallFunctionObjArgs(expr, callables, NULL);
          Py_DECREF(expr);
          if (grid_size) {
            prg->py_global = PyTuple_GetItem(grid_size, 0);
            prg->py_local = PyTuple_GetItem(grid_size, 1);
            prg->ndim =
                MAX(PyTuple_Size(prg->py_global), PyTuple_Size(prg->py_local));
            err = 0;
          }
        }
      }
      Py_DECREF(callables);
    }
  }
  if (err) {
    return set_log(NOMP_LOOPY_GET_GRIDSIZE_FAILED, NOMP_ERROR,
                   ERR_STR_LOOPY_GRIDSIZE_FAILED);
  }
  return 0;
}

static int py_eval_grid_size_aux(size_t *out, PyObject *grid, unsigned dim,
                                 PyObject *evaluate, PyObject *dict) {
  PyObject *py_dim = PyTuple_GetItem(grid, dim);
  int err = 1;
  if (py_dim) {
    PyObject *rslt = PyObject_CallFunctionObjArgs(evaluate, py_dim, dict, NULL);
    if (rslt) {
      out[dim] = PyLong_AsLong(rslt);
      Py_DECREF(rslt), err = 0;
    }
  }

  return err;
}

int py_eval_grid_size(struct prog *prg, PyObject *dict) {
  // If the expressions are not NULL, iterate through them and evaluate with
  // pymbolic. Also, we should calculate and store a hash of the dict that
  // is passed. If the hash is the same, no need of re-evaluating the grid
  // size.
  for (unsigned i = 0; i < prg->ndim; i++)
    prg->global[i] = prg->local[i] = 1;

  int err = 1;
  if (prg->py_global && prg->py_local) {
    PyObject *mapper = PyImport_ImportModule("pymbolic.mapper.evaluator");
    if (mapper) {
      PyObject *evaluate = PyObject_GetAttrString(mapper, "evaluate");
      Py_DECREF(mapper);
      if (evaluate) {
        err = 0;
        // Iterate through grid sizes, evaluate and set `global` and `local`
        // sizes respectively.
        for (unsigned i = 0; i < PyTuple_Size(prg->py_global); i++)
          err |= py_eval_grid_size_aux(prg->global, prg->py_global, i, evaluate,
                                       dict);
        for (unsigned i = 0; i < PyTuple_Size(prg->py_local); i++)
          err |= py_eval_grid_size_aux(prg->local, prg->py_local, i, evaluate,
                                       dict);
        Py_DECREF(evaluate);
      }
    }
  }

  if (err) {
    return set_log(NOMP_LOOPY_EVAL_GRIDSIZE_FAILED, NOMP_ERROR,
                   "libnomp was unable to evaluate the kernel launch "
                   "parameters using pymbolic.");
  }
  return 0;
}

#undef MAX
