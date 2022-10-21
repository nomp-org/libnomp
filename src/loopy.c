#include "nomp-impl.h"

static const char *loopy_api = "loopy_api";
static const char *c_to_loopy = "c_to_loopy";

void py_print(PyObject *obj) {
  PyObject *repr = PyObject_Repr(obj);
  PyObject *py_str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
  const char *str = PyBytes_AS_STRING(py_str);
  printf("%s", str);
  Py_XDECREF(repr), Py_XDECREF(py_str);
}

int py_append_to_sys_path(const char *path) {
  int err = NOMP_PY_APPEND_PATH_ERROR;

  PyObject *py_sys = PyImport_ImportModule("sys");
  if (py_sys) {
    PyObject *py_path = PyObject_GetAttrString(py_sys, "path");
    if (py_path) {
      PyObject *py_str = PyUnicode_FromString(path);
      PyList_Append(py_path, py_str);
      Py_DECREF(py_path), Py_XDECREF(py_str), err = 0;
    }
    Py_DECREF(py_sys);
  }

  return err;
}

int py_c_to_loopy(PyObject **knl, const char *src, const char *backend) {
  int err = NOMP_LOOPY_CONVERSION_ERROR;

  PyObject *lpy_api = PyUnicode_FromString(loopy_api);
  if (lpy_api) {
    PyObject *module = PyImport_Import(lpy_api);
    if (module) {
      PyObject *c_to_lpy = PyObject_GetAttrString(module, c_to_loopy);
      if (c_to_lpy) {
        PyObject *py_src = PyUnicode_FromString(src);
        PyObject *py_backend = PyUnicode_FromString(backend);
        *knl = PyObject_CallFunctionObjArgs(c_to_lpy, py_src, py_backend, NULL);
        if (*knl)
          err = 0;
        Py_XDECREF(py_src), Py_XDECREF(py_backend), Py_DECREF(c_to_lpy);
      }
      Py_DECREF(module);
    }
    Py_DECREF(lpy_api);
  }
  if (err)
    return nomp_set_log(NOMP_LOOPY_CONVERSION_ERROR, NOMP_ERROR,
                        ERR_STR_LOOPY_CONVERSION_ERROR);
  return err;
}

int py_user_callback(PyObject **knl, const char *file, const char *func) {
  int err = NOMP_USER_CALLBACK_NOT_FOUND;

  if (*knl && file && func) {
    PyObject *py_file = PyUnicode_FromString(file);
    if (py_file) {
      PyObject *module = PyImport_Import(py_file);
      if (module) {
        PyObject *py_func = PyObject_GetAttrString(module, func);
        if (py_func && PyCallable_Check(py_func)) {
          err = NOMP_USER_CALLBACK_FAILURE;
          PyObject *tknl = PyObject_CallFunctionObjArgs(py_func, *knl, NULL);
          if (tknl)
            Py_DECREF(*knl), *knl = tknl, tknl = NULL, err = 0;
          Py_DECREF(py_func);
        }
        Py_DECREF(module);
      }
      Py_DECREF(py_file);
    }
  }
  if (err) {
    if (err == NOMP_USER_CALLBACK_NOT_FOUND) {
      err =
          nomp_set_log(err, NOMP_ERROR, ERR_STR_USER_CALLBACK_NOT_FOUND, file);
    } else {
      err = nomp_set_log(err, NOMP_ERROR, ERR_STR_USER_CALLBACK_FAILURE, func);
    }
  }
  return err;
}

int py_get_knl_name_and_src(char **name, char **src, PyObject *knl) {
  // FIXME: This should be an error defined in nomp.h
  int err = 1;

  if (knl) {
    err = NOMP_LOOPY_KNL_NAME_NOT_FOUND;
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
          Py_ssize_t size;
          const char *name_ = PyUnicode_AsUTF8AndSize(py_name, &size);
          *name = (char *)calloc(size + 1, sizeof(char));
          strncpy(*name, name_, size + 1);
          Py_DECREF(py_name), err = 0;
        }
        Py_XDECREF(entry), Py_DECREF(iter);
      }
      Py_DECREF(epts);
    }
    if (err) {
      PyErr_Print();
      return nomp_set_log(NOMP_LOOPY_KNL_NAME_NOT_FOUND, NOMP_ERROR,
                          ERR_STR_LOOPY_KNL_NAME_NOT_FOUND, *name);
    }
    return_on_err(err);

    // Get the kernel source
    err = NOMP_LOOPY_CODEGEN_FAILED;
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
              Py_ssize_t size;
              const char *src_ = PyUnicode_AsUTF8AndSize(py_src, &size);
              *src = (char *)calloc(size + 1, sizeof(char));
              strncpy(*src, src_, size + 1);
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
  }
  if (err) {
      PyErr_Print();
      return nomp_set_log(NOMP_LOOPY_CODEGEN_FAILED, NOMP_ERROR,
                          ERR_STR_LOOPY_CODEGEN_FAILED, *name);
  }
  return err;
}

int py_get_grid_size(struct prog *prg, PyObject *knl) {
  int err = NOMP_GET_GRIDSIZE_FAILED;

  if (knl) {
    // knl.callables_table
    PyObject *callables = PyObject_GetAttrString(knl, "callables_table");
    if (callables) {
      // knl.default_entrypoint.get_grid_size_upper_bounds_as_exprs
      PyObject *entry = PyObject_GetAttrString(knl, "default_entrypoint");
      if (entry) {
        PyObject *expr = PyObject_GetAttrString(
            entry, "get_grid_size_upper_bounds_as_exprs");
        if (expr) {
          PyObject *grid_size =
              PyObject_CallFunctionObjArgs(expr, callables, NULL);
          if (grid_size) {
            prg->py_global = PyTuple_GetItem(grid_size, 0);
            prg->py_local = PyTuple_GetItem(grid_size, 1);
            prg->ndim = PyTuple_Size(prg->py_global);
            if (PyTuple_Size(prg->py_local) > prg->ndim)
              prg->ndim = PyTuple_Size(prg->py_local);
            err = 0;
          }
          Py_DECREF(expr);
        }
        Py_DECREF(entry);
      }
      Py_DECREF(callables);
    }
  }
  if (err) {
      PyErr_Print();
      return nomp_set_log(NOMP_GET_GRIDSIZE_FAILED, NOMP_ERROR,
                          ERR_STR_LOOPY_GRIDSIZE_FAILED);
  }
  return err;
}

static int py_eval_grid_size_aux(size_t *out, PyObject *grid, unsigned dim,
                                 PyObject *evaluate, PyObject *dict) {
  // FIXME: This should be an error defined in nomp.h
  int err = 1;

  PyObject *py_dim = PyTuple_GetItem(grid, dim);
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

  int err = 0;
  if (prg->py_global && prg->py_local) {
    PyObject *mapper = PyImport_ImportModule("pymbolic.mapper.evaluator");
    if (mapper) {
      PyObject *evaluate = PyObject_GetAttrString(mapper, "evaluate");
      if (evaluate) {
        // Iterate through grid sizes, evaluate and set `global` and `local`
        // sizes respectively.
        for (unsigned i = 0; i < PyTuple_Size(prg->py_global); i++)
          err |= py_eval_grid_size_aux(prg->global, prg->py_global, i, evaluate,
                                       dict);
        for (unsigned i = 0; i < PyTuple_Size(prg->py_local); i++)
          err |= py_eval_grid_size_aux(prg->local, prg->py_local, i, evaluate,
                                       dict);
        Py_DECREF(evaluate);
        err = (err != 0) * NOMP_EVAL_GRIDSIZE_FAILED;
      }
      Py_DECREF(mapper);
    }
    if (err) {
      PyErr_Print();
      return nomp_set_log(NOMP_EVAL_GRIDSIZE_FAILED, NOMP_ERROR,
                          ERR_STR_GRIDSIZE_CALCULATION_FAILED);
    }
  }

  return err;
}
