#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "nomp-impl.h"

static int get_loopy_knl_name(char **name, PyObject *pKnl) {
  // Get the kernel name from loopy kernel
  int err = 1;
  PyObject *pEntrypts = PyObject_GetAttrString(pKnl, "entrypoints");
  if (pEntrypts) {
    Py_ssize_t len = PySet_Size(pEntrypts);
    // Iterator C API: https://docs.python.org/3/c-api/iter.html
    PyObject *pIter = PyObject_GetIter(pEntrypts);
    if (pIter) {
      PyObject *pEntry = PyIter_Next(pIter);
      PyObject *pKnlName = PyObject_Str(pEntry);
      if (pKnlName) {
        Py_ssize_t size;
        const char *name_ = PyUnicode_AsUTF8AndSize(pKnlName, &size);
        *name = (char *)calloc(size + 1, sizeof(char));
        strncpy(*name, name_, size);
        Py_DECREF(pKnlName), err = 0;
      }
      Py_XDECREF(pEntry), Py_DECREF(pIter);
    }
    Py_DECREF(pEntrypts);
  }

  if (err)
    PyErr_Print();
  return err;
}

int py_append_to_sys_path(const char *path) {
  PyObject *pSys = PyImport_ImportModule("sys");
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

  if (err)
    PyErr_Print();
  return err;
}

int py_user_callback(struct knl *knl, const char *c_src, const char *file,
                     const char *func) {
  // Create the loopy kernel based on `c_src`.
  const char *py_module = "c_to_loopy";
  int err = NOMP_C_TO_LOOPY_CONVERSION_ERROR;
  PyObject *pFile = PyUnicode_FromString(py_module),
           *pModule = PyImport_Import(pFile), *pKnl = NULL;
  Py_XDECREF(pFile);
  if (pModule) {
    PyObject *pFunc = PyObject_GetAttrString(pModule, py_module);
    if (pFunc && PyCallable_Check(pFunc)) {
      PyObject *pStr = PyUnicode_FromString(c_src);
      pKnl = PyObject_CallFunctionObjArgs(pFunc, pStr, NULL);
      err = (pKnl == NULL);
      Py_XDECREF(pStr), Py_DECREF(pFunc);
    }
    Py_DECREF(pModule);
  }
  if (err) {
    PyErr_Print();
    return err;
  }

  // TODO: Apply Domain specific transformations

  // Call the user callback if present
  if (pKnl && file && func) {
    err = NOMP_USER_CALLBACK_NOT_FOUND;
    PyObject *pFile = PyUnicode_FromString(file),
             *pModule = PyImport_Import(pFile), *pTransformedKnl = NULL;
    if (pModule) {
      PyObject *pFunc = PyObject_GetAttrString(pModule, func);
      if (pFunc && PyCallable_Check(pFunc)) {
        err = NOMP_USER_CALLBACK_FAILURE;
        pTransformedKnl = PyObject_CallFunctionObjArgs(pFunc, pKnl, NULL);
        if (pTransformedKnl) {
          Py_DECREF(pKnl), pKnl = pTransformedKnl;
          pTransformedKnl = NULL, err = 0;
        }
        Py_DECREF(pFunc);
      }
      Py_DECREF(pModule);
    }
    Py_XDECREF(pFile);
  }
  if (err) {
    PyErr_Print();
    return err;
  }

  // Get grid size, OpenCL source, etc from transformed kernel
  if (pKnl) {
    err = NOMP_CODEGEN_FAILED;
    if (get_loopy_knl_name(&knl->name, pKnl))
      return err;
    // FIXME: This should only be done once
    PyObject *pLoopy = PyImport_ImportModule("loopy");
    if (pLoopy) {
      PyObject *pGenerateCodeV2 =
          PyObject_GetAttrString(pLoopy, "generate_code_v2");
      if (pGenerateCodeV2) {
        PyObject *pArgs = PyTuple_New(1);
        PyTuple_SetItem(pArgs, 0, pKnl);
        PyObject *pCode = PyObject_CallObject(pGenerateCodeV2, pArgs);
        if (pCode) {
          PyObject *pDeviceCode = PyObject_GetAttrString(pCode, "device_code");
          if (pDeviceCode) {
            PyObject *pSrc = PyObject_CallObject(pDeviceCode, PyTuple_New(0));
            if (pSrc) {
              Py_ssize_t size;
              const char *src = PyUnicode_AsUTF8AndSize(pSrc, &size);
              knl->src = (char *)calloc(size + 1, sizeof(char));
              memcpy(knl->src, src, sizeof(char) * size);
              Py_DECREF(pSrc), err = 0;
            }
            Py_DECREF(pDeviceCode);
          }
          Py_DECREF(pCode);
        }
        Py_XDECREF(pArgs);
        Py_DECREF(pGenerateCodeV2);
      }
      Py_DECREF(pLoopy);
    }
    Py_DECREF(pKnl);
  }

  if (err)
    PyErr_Print();
  return err;
}
