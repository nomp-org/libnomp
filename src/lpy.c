#include "nomp-impl.h"

static const char *module_loopy_api = "loopy_api";
static const char *module_reduction = "reduction";

static const char *c_to_loopy = "c_to_loopy";
static const char *get_knl_src = "get_knl_src";
static const char *get_knl_name = "get_knl_name";
static const char *realize_reduction = "realize_reduction";

void nomp_py_print(const char *msg, PyObject *obj) {
  PyObject *repr = PyObject_Repr(obj);
  PyObject *py_str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
  const char *str = PyBytes_AS_STRING(py_str);
  printf("%s: %s\n", msg, str);
  Py_XDECREF(repr), Py_XDECREF(py_str);
}

int nomp_py_append_to_sys_path(const char *path) {
  int err = 1;
  PyObject *sys = PyImport_ImportModule("sys");
  if (sys) {
    PyObject *ppath = PyObject_GetAttrString(sys, "path");
    Py_DECREF(sys);
    if (ppath) {
      PyObject *pstr = PyUnicode_FromString(path);
      err = PyList_Append(ppath, pstr);
      Py_DECREF(ppath), Py_XDECREF(pstr);
    }
  }
  if (err) {
    return nomp_set_log(NOMP_PY_CALL_FAILURE, NOMP_ERROR,
                        "Appending path \"%s\" to the sys.path failed.", path);
  }

  return 0;
}

static int py_realize_reduction(PyObject **knl, const char *backend) {
  int err = 1;
  PyObject *reduction = PyUnicode_FromString(module_reduction);
  if (reduction) {
    PyObject *module = PyImport_Import(reduction);
    if (module) {
      PyObject *rr = PyObject_GetAttrString(module, realize_reduction);
      if (rr) {
        PyObject *pbackend = PyUnicode_FromString(backend);
        PyObject *result =
            PyObject_CallFunctionObjArgs(rr, *knl, pbackend, NULL);
        if (result)
          Py_DECREF(*knl), *knl = result, err = 0;
        Py_XDECREF(pbackend), Py_DECREF(rr);
      }
      Py_DECREF(module);
    }
    Py_DECREF(reduction);
  }
  if (err) {
    return nomp_set_log(NOMP_PY_CALL_FAILURE, NOMP_ERROR,
                        "Call to realize_reduction failed.");
  }

  return 0;
}

int py_c_to_loopy(PyObject **knl, const char *src, const char *backend,
                  int reduction_index) {
  int err = 1;
  PyObject *lpy_api = PyUnicode_FromString(module_loopy_api);
  if (lpy_api) {
    PyObject *module = PyImport_Import(lpy_api);
    if (module) {
      PyObject *c_to_lpy = PyObject_GetAttrString(module, c_to_loopy);
      if (c_to_lpy) {
        PyObject *psrc = PyUnicode_FromString(src);
        PyObject *pbackend = PyUnicode_FromString(backend);
        PyObject *pindex = PyLong_FromLong(reduction_index);
        *knl = PyObject_CallFunctionObjArgs(c_to_lpy, psrc, pbackend, pindex,
                                            NULL);
        Py_XDECREF(psrc), Py_XDECREF(pbackend), Py_XDECREF(pindex);
        Py_DECREF(c_to_lpy), err = (*knl == NULL);
      }
      Py_DECREF(module);
    }
    Py_DECREF(lpy_api);
  }
  if (err) {
    return nomp_set_log(NOMP_LOOPY_CONVERSION_FAILURE, NOMP_ERROR,
                        "C to Loopy conversion failed.\n");
  }

  if (reduction_index >= 0)
    nomp_check(py_realize_reduction(knl, backend));

  return 0;
}

int py_set_annotate_func(PyObject **annotate_func, const char *path_) {
  // Find file and function from path.
  char *path = strndup(path_, PATH_MAX + NOMP_MAX_IDENT_SIZE);
  char *file = strtok(path, "::"), *func = strtok(NULL, "::");
  if (file == NULL || path == NULL) {
    nomp_free(path);
    return 0;
  }

  nomp_check(nomp_check_py_script_path(file));

  int err = 1;
  PyObject *pfile = PyUnicode_FromString(file);
  if (pfile) {
    PyObject *module = PyImport_Import(pfile);
    Py_DECREF(pfile);
    if (module) {
      PyObject *pfunc = PyObject_GetAttrString(module, func);
      Py_DECREF(module);
      if (pfunc && PyCallable_Check(pfunc))
        Py_XDECREF(*annotate_func), *annotate_func = pfunc, err = 0;
    }
  }
  if (err) {
    err = nomp_set_log(
        NOMP_PY_CALL_FAILURE, NOMP_ERROR,
        "Failed to find annotate function \"%s\" in file \"%s\".", func, file);
  }

  nomp_free(path);
  return err;
}

int py_apply_annotations(PyObject **knl, PyObject *func, PyObject *annts) {
  if (knl && *knl && func) {
    if (PyCallable_Check(func)) {
      PyObject *tknl = PyObject_CallFunctionObjArgs(func, *knl, annts, NULL);
      if (tknl)
        Py_DECREF(*knl), *knl = tknl;
    }
  }

  return 0;
}

int nomp_py_user_transform(PyObject **knl, const char *file, const char *func) {
  // If either file, or func are NULL, we don't have to do anything:
  if (file == NULL || func == NULL)
    return 0;

  int err = 1;
  PyObject *pfile = PyUnicode_FromString(file);
  if (knl && *knl && pfile) {
    PyObject *module = PyImport_Import(pfile);
    Py_DECREF(pfile);
    if (module) {
      PyObject *pfunc = PyObject_GetAttrString(module, func);
      Py_DECREF(module);
      if (pfunc && PyCallable_Check(pfunc)) {
        PyObject *tknl = PyObject_CallFunctionObjArgs(pfunc, *knl, NULL);
        if (tknl)
          Py_DECREF(*knl), *knl = tknl, err = 0;
        Py_DECREF(pfunc);
      }
    }
  }
  if (err) {
    return nomp_set_log(
        NOMP_PY_CALL_FAILURE, NOMP_ERROR,
        "Failed to call user transform function: \"%s\" in file: \"%s\".", func,
        file);
  }

  return 0;
}

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
        return nomp_set_log(NOMP_LOOPY_KNL_NAME_NOT_FOUND, NOMP_ERROR,
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
      return nomp_set_log(
          NOMP_LOOPY_CODEGEN_FAILURE, NOMP_ERROR,
          "Backend code generation from loopy kernel \"%s\" failed.", *name);
    }
  }

  return 0;
}

int nomp_py_get_grid_size(struct nomp_prog *prg, PyObject *knl) {
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
    return nomp_set_log(NOMP_LOOPY_GET_GRIDSIZE_FAILURE, NOMP_ERROR,
                        "Unable to get grid sizes from loopy kernel.");
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

int nomp_py_eval_grid_size(struct nomp_prog *prg) {
  // If the expressions are not NULL, iterate through them and evaluate with
  // pymbolic. Also, we should calculate and store a hash of the dict that
  // is passed. If the hash is the same, no need of re-evaluating the grid
  // size.
  for (unsigned i = 0; i < 3; i++)
    prg->global[i] = prg->local[i] = 1;

  int err = 1;
  if (prg->py_global && prg->py_local) {
    PyObject *mapper = PyImport_ImportModule("pymbolic.mapper.evaluator");
    if (mapper) {
      PyObject *evaluate = PyObject_GetAttrString(mapper, "evaluate");
      if (evaluate) {
        err = 0;
        // Iterate through grid sizes, evaluate and set `global` and `local`
        // sizes respectively.
        for (unsigned i = 0; i < PyTuple_Size(prg->py_global); i++)
          err |= py_eval_grid_size_aux(prg->global, prg->py_global, i, evaluate,
                                       prg->py_dict);
        for (unsigned i = 0; i < PyTuple_Size(prg->py_local); i++)
          err |= py_eval_grid_size_aux(prg->local, prg->py_local, i, evaluate,
                                       prg->py_dict);
        Py_DECREF(evaluate);
      }
      Py_DECREF(mapper);
    }
  }
  if (err) {
    return nomp_set_log(NOMP_LOOPY_EVAL_GRIDSIZE_FAILURE, NOMP_ERROR,
                        "libnomp was unable to evaluate the kernel launch "
                        "parameters using pymbolic.");
  }

  return 0;
}
