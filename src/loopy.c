#include "nomp-impl.h"

int py_append_to_sys_path(const char *path) {
  PyObject *pSys = PyImport_ImportModule("sys");
  // FIXME: Err should be an nomp error defined in `nomp.h`
  int err = 1;
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
  // Create the loopy kernel based on `c_src`.
  int err = NOMP_LOOPY_CONVERSION_ERROR;
  PyObject *pModuleStr = PyUnicode_FromString(py_module),
           *pModule = PyImport_Import(pModuleStr);
  Py_XDECREF(pModuleStr);
  if (pModule) {
    PyObject *pFunc = PyObject_GetAttrString(pModule, py_func);
    if (pFunc) {
      PyObject *pStr = PyUnicode_FromString(c_src);
      PyObject *pBackend = PyUnicode_FromString(backend);
      if ((*pKnl = PyObject_CallFunctionObjArgs(pFunc, pStr, pBackend, NULL)))
        err = 0;
      Py_XDECREF(pStr), Py_XDECREF(pBackend), Py_DECREF(pFunc);
    }
    Py_DECREF(pModule);
  }
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

int py_get_grid_size(int *ndim, size_t *global, size_t *local, PyObject *pKnl,
                     PyObject *pDict) {
  // Intiialize global and local sizes to 1
  global[0] = global[1] = global[2] = 1;
  local[0] = local[1] = local[2] = 1;

  if (pKnl) {
    // Get global and local grid size as experssions from loopy
    int err = NOMP_LOOPY_GRIDSIZE_FAILED;
    PyObject *pGridSize = NULL;
    // knl.callables_table
    PyObject *pCallablesTable = PyObject_GetAttrString(pKnl, "callables_table");
    if (pCallablesTable) {
      // knl.default_entrypoint.get_grid_size_upper_bounds_as_exprs
      PyObject *pDefaultEntryPoint =
          PyObject_GetAttrString(pKnl, "default_entrypoint");
      if (pDefaultEntryPoint) {
        PyObject *pGridSizeUboundAsExpr = PyObject_GetAttrString(
            pDefaultEntryPoint, "get_grid_size_upper_bounds_as_exprs");
        if (pGridSizeUboundAsExpr) {
          pGridSize = PyObject_CallFunctionObjArgs(pGridSizeUboundAsExpr,
                                                   pCallablesTable, NULL);
          Py_DECREF(pGridSizeUboundAsExpr), err = 0;
        }
        Py_DECREF(pDefaultEntryPoint);
      }
      Py_DECREF(pCallablesTable);
    }
    if (err) {
      PyErr_Print();
      return err;
    }

    // If the expressions are not NULL, iterate through them and evaluate with
    // pymbolic
    err = NOMP_GRIDSIZE_CALCULATION_FAILED;
    if (pGridSize) {
      assert(PyTuple_Check(pGridSize));
      assert(PyTuple_Size(pGridSize) == 2);
      PyObject *pEvaluator = PyImport_ImportModule("pymbolic.mapper.evaluator");
      if (pEvaluator) {
        PyObject *pEvaluate = PyObject_GetAttrString(pEvaluator, "evaluate");
        if (pEvaluate) {
          // Iterate through grid sizes, evaluate and set `global` and `local`
          // sizes respectively.
          PyObject *pGlobal = PyTuple_GetItem(pGridSize, 0);
          assert(PyTuple_Check(pGlobal));
          *ndim = PyTuple_Size(pGlobal);
          for (int i = 0; i < PyTuple_Size(pGlobal); i++) {
            PyObject *pDim = PyTuple_GetItem(pGlobal, i);
            PyObject *pResult =
                PyObject_CallFunctionObjArgs(pEvaluate, pDim, pDict, NULL);
            if (pResult)
              global[i] = PyLong_AsLong(pResult);
          }

          PyObject *pLocal = PyTuple_GetItem(pGridSize, 1);
          assert(PyTuple_Check(pLocal));
          for (int i = 0; i < PyTuple_Size(pLocal); i++) {
            PyObject *pDim = PyTuple_GetItem(pLocal, i);
            PyObject *pResult =
                PyObject_CallFunctionObjArgs(pEvaluate, pDim, pDict, NULL);
            if (pResult)
              local[i] = PyLong_AsLong(pResult);
          }
          Py_DECREF(pEvaluate), err = 0;
        }
        Py_DECREF(pEvaluator);
      }
      Py_DECREF(pGridSize);
    }
    if (err) {
      PyErr_Print();
      return err;
    }
  }

  return 0;
}
