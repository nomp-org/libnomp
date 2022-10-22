#include "nomp-impl.h"

static const char *py_module = "loopy_api";
static const char *py_func = "c_to_loopy";

void py_print(PyObject *obj) {
  PyObject *repr = PyObject_Repr(obj);
  PyObject *str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
  const char *bytes = PyBytes_AS_STRING(str);
  printf("%s", bytes);
  Py_XDECREF(repr);
  Py_XDECREF(str);
}

int py_append_to_sys_path(const char *path) {
  PyObject *pSys = PyImport_ImportModule("sys");
  int err = NOMP_PY_APPEND_PATH_ERROR;
  if (pSys) {
    PyObject *pPath = PyObject_GetAttrString(pSys, "path");
    Py_DECREF(pSys);
    if (pPath) {
      PyObject *pStr = PyUnicode_FromString(path);
      PyList_Append(pPath, pStr);
      Py_DECREF(pPath), Py_XDECREF(pStr), err = 0;
    }
  }
  if (err) {
    PyErr_Print();
    return err;
  }
  return 0;
}

int py_c_to_loopy(PyObject **pKnl, const char *c_src, const char *backend) {
  int err = NOMP_LOOPY_CONVERSION_ERROR;
  PyObject *pModuleStr = PyUnicode_FromString(py_module),
           *pModule = PyImport_Import(pModuleStr);
  if (pModule) {
    PyObject *pFunc = PyObject_GetAttrString(pModule, py_func);
    if (pFunc) {
      PyObject *pStr = PyUnicode_FromString(c_src);
      PyObject *pBackend = PyUnicode_FromString(backend);
      *pKnl = PyObject_CallFunctionObjArgs(pFunc, pStr, pBackend, NULL);
      if (*pKnl)
        err = 0;
      Py_XDECREF(pStr), Py_XDECREF(pBackend), Py_DECREF(pFunc);
    }
    Py_DECREF(pModule);
  }
  Py_XDECREF(pModuleStr);
  return err;
}

int py_user_callback(PyObject **pKnl, const char *file, const char *func) {
  // Call the user callback if present
  int err = NOMP_USER_CALLBACK_NOT_FOUND;
  if (*pKnl && file && func) {
    PyObject *pFile = PyUnicode_FromString(file),
             *pModule = PyImport_Import(pFile), *pTransformedKnl = NULL;
    if (pModule) {
      PyObject *pFunc = PyObject_GetAttrString(pModule, func);
      if (pFunc && PyCallable_Check(pFunc)) {
        err = NOMP_USER_CALLBACK_FAILURE;
        pTransformedKnl = PyObject_CallFunctionObjArgs(pFunc, *pKnl, NULL);
        if (pTransformedKnl) {
          Py_DECREF(*pKnl), *pKnl = pTransformedKnl;
          pTransformedKnl = NULL, err = 0;
        }
        Py_DECREF(pFunc);
      }
      Py_DECREF(pModule);
    }
    Py_XDECREF(pFile);
  }
  return err;
}

int py_get_knl_name_and_src(char **name, char **src, PyObject *pKnl) {
  if (pKnl) {
    // Get the kernel name from loopy kernel
    int err = NOMP_LOOPY_KNL_NAME_NOT_FOUND;
    PyObject *pEntrypts = PyObject_GetAttrString(pKnl, "entrypoints");
    if (pEntrypts) {
      Py_ssize_t len = PySet_Size(pEntrypts);
      // FIXME: This doesn't require iterator API
      // Iterator C API: https://docs.python.org/3/c-api/iter.html
      PyObject *pIter = PyObject_GetIter(pEntrypts);
      if (pIter) {
        PyObject *pEntry = PyIter_Next(pIter);
        PyObject *pKnlName = PyObject_Str(pEntry);
        if (pKnlName) {
          Py_ssize_t size;
          const char *name_ = PyUnicode_AsUTF8AndSize(pKnlName, &size);
          *name = (char *)calloc(size + 1, sizeof(char));
          strncpy(*name, name_, size + 1);
          Py_DECREF(pKnlName), err = 0;
        }
        Py_XDECREF(pEntry), Py_DECREF(pIter);
      }
      Py_DECREF(pEntrypts);
    }
    if (err) {
      PyErr_Print();
      return err;
    }

    // Get the kernel source
    err = NOMP_LOOPY_CODEGEN_FAILED;
    PyObject *pLoopy = PyImport_ImportModule("loopy");
    if (pLoopy) {
      PyObject *pGenerateCodeV2 =
          PyObject_GetAttrString(pLoopy, "generate_code_v2");
      if (pGenerateCodeV2) {
        PyObject *pCode =
            PyObject_CallFunctionObjArgs(pGenerateCodeV2, pKnl, NULL);
        if (pCode) {
          PyObject *pDeviceCode = PyObject_GetAttrString(pCode, "device_code");
          if (pDeviceCode) {
            PyObject *pSrc = PyObject_CallFunctionObjArgs(pDeviceCode, NULL);
            if (pSrc) {
              Py_ssize_t size;
              const char *src_ = PyUnicode_AsUTF8AndSize(pSrc, &size);
              *src = (char *)calloc(size + 1, sizeof(char));
              strncpy(*src, src_, size + 1);
              Py_DECREF(pSrc), err = 0;
            }
            Py_DECREF(pDeviceCode);
          }
          Py_DECREF(pCode);
        }
        Py_DECREF(pGenerateCodeV2);
      }
      Py_DECREF(pLoopy);
    }
    if (err) {
      PyErr_Print();
      return err;
    }
  }
  return 0;
}

int py_get_grid_size(struct prog *prg, PyObject *knl) {
  int err = NOMP_GET_GRIDSIZE_FAILED;
  if (knl) {
    // knl.callables_table
    PyObject *py_callables = PyObject_GetAttrString(knl, "callables_table");
    if (py_callables) {
      // knl.default_entrypoint.get_grid_size_upper_bounds_as_exprs
      PyObject *py_entry = PyObject_GetAttrString(knl, "default_entrypoint");
      if (py_entry) {
        PyObject *py_expr = PyObject_GetAttrString(
            py_entry, "get_grid_size_upper_bounds_as_exprs");
        if (py_expr) {
          PyObject *py_grid_size =
              PyObject_CallFunctionObjArgs(py_expr, py_callables, NULL);
          if (py_grid_size) {
            prg->py_global = PyTuple_GetItem(py_grid_size, 0);
            prg->py_local = PyTuple_GetItem(py_grid_size, 1);
            prg->ndim = PyTuple_Size(prg->py_global);
            if (PyTuple_Size(prg->py_local) > prg->ndim)
              prg->ndim = PyTuple_Size(prg->py_local);
            err = 0;
          }
          Py_DECREF(py_expr);
        }
        Py_DECREF(py_entry);
      }
      Py_DECREF(py_callables);
    }
  }
  return err;
}

static int py_eval_grid_size_aux(size_t *out, PyObject *py_grid, unsigned dim,
                                 PyObject *py_evaluate, PyObject *py_dict) {
  PyObject *py_dim = PyTuple_GetItem(py_grid, dim);
  int err = 1;
  if (py_dim) {
    PyObject *py_result =
        PyObject_CallFunctionObjArgs(py_evaluate, py_dim, py_dict, NULL);
    if (py_result) {
      out[dim] = PyLong_AsLong(py_result);
      Py_DECREF(py_result), err = 0;
    }
  }
  return err;
}

int py_eval_grid_size(struct prog *prg, PyObject *py_dict) {
  // If the expressions are not NULL, iterate through them and evaluate with
  // pymbolic. Also, we should calculate and store a hash of the py_dict that
  // is passed. If the hash is the same, no need of re-evaluating the grid
  // size.
  for (unsigned i = 0; i < prg->ndim; i++)
    prg->global[i] = prg->local[i] = 1;

  int err = 0;
  if (prg->py_global && prg->py_local) {
    PyObject *py_mapper = PyImport_ImportModule("pymbolic.mapper.evaluator");
    if (py_mapper) {
      PyObject *py_evaluate = PyObject_GetAttrString(py_mapper, "evaluate");
      if (py_evaluate) {
        // Iterate through grid sizes, evaluate and set `global` and `local`
        // sizes respectively.
        for (unsigned i = 0; i < PyTuple_Size(prg->py_global); i++)
          err |= py_eval_grid_size_aux(prg->global, prg->py_global, i,
                                       py_evaluate, py_dict);
        for (unsigned i = 0; i < PyTuple_Size(prg->py_local); i++)
          err |= py_eval_grid_size_aux(prg->local, prg->py_local, i,
                                       py_evaluate, py_dict);
        Py_DECREF(py_evaluate);
        err = (err != 0) * NOMP_EVAL_GRIDSIZE_FAILED;
      }
      Py_DECREF(py_mapper);
    }
  }
  return err;
}
