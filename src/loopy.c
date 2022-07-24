#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "nomp-impl.h"

static PyObject *call_pickle_loads(PyObject *pDecodedBinStr) {
  PyObject *pPickle = PyImport_ImportModule("pickle"), *pLpyKnl = NULL;
  if (pPickle) {
    PyObject *pLoads = PyObject_GetAttrString(pPickle, "loads");
    if (pLoads && PyCallable_Check(pLoads)) {
      PyObject *pArgs = PyTuple_New(1);
      PyTuple_SetItem(pArgs, 0, pDecodedBinStr);
      pLpyKnl = PyObject_CallObject(pLoads, pArgs);
      Py_DECREF(pArgs);
      Py_DECREF(pLoads);
    }
    Py_DECREF(pPickle);
  }
  return pLpyKnl;
}

static PyObject *call_base64_decode(PyObject *pBinStr) {
  PyObject *pBase64 = PyImport_ImportModule("base64"), *pDecodedBinStr = NULL;
  if (pBase64) {
    PyObject *pDecode = PyObject_GetAttrString(pBase64, "b64decode");
    if (pDecode && PyCallable_Check(pDecode)) {
      PyObject *pArgs = PyTuple_New(1);
      PyTuple_SetItem(pArgs, 0, pBinStr);
      pDecodedBinStr = PyObject_CallObject(pDecode, pArgs);
      Py_DECREF(pArgs);
      Py_DECREF(pDecode);
    }
    Py_DECREF(pBase64);
  }

  return pDecodedBinStr;
}

static char *get_loopy_knl_name(PyObject *pKnl) {
  // Get the kernel name from loopy kernel
  PyObject *pKnlName = NULL;
  PyObject *pEntrypts = PyObject_GetAttrString(pKnl, "entrypoints");
  if (pEntrypts && PyFrozenSet_Check(pEntrypts)) {
    Py_ssize_t len = PySet_Size(pEntrypts);
    assert(len == 1);

    // Iterator C API: https://docs.python.org/3/c-api/iter.html
    PyObject *pIter = PyObject_GetIter(pEntrypts);
    if (pIter) {
      PyObject *pEntry = PyIter_Next(pIter);
      pKnlName = PyObject_Str(pEntry);
      Py_XDECREF(pEntry);
      Py_DECREF(pIter);
    }
    Py_DECREF(pEntrypts);
  }

  char *name = NULL;
  if (pKnlName) {
    const char *name_ = PyUnicode_AsUTF8(pKnlName);
    size_t len = strlen(name_) + 1;
    name = (char *)calloc(len, sizeof(char));
    strncpy(name, name_, len);
    Py_XDECREF(pKnlName);
  }

  return name;
}

static int append_to_sys_path(const char *path) {
  PyObject *pSys = PyImport_ImportModule("sys");
  if (pSys) {
    PyObject *pPath = PyObject_GetAttrString(pSys, "path");
    Py_DECREF(pSys);
    if (pPath) {
      PyObject *pStr = PyUnicode_FromString(path);
      PyList_Append(pPath, pStr);
      Py_DECREF(pPath), Py_XDECREF(pStr);
      return 0;
    }
  }
  return 1;
}

int py_user_callback(struct knl *knl, const char *c_src, const char *file,
                     const char *func) {
  if (!Py_IsInitialized()) {
    Py_Initialize();
    append_to_sys_path(".");
  }

  // Create the loop kernel based on `c_src`.
  PyObject *pKnl = NULL;

  // There should be a better way to figure the installation
  // path based on the shared library path
  int err = NOMP_INSTALL_DIR_NOT_FOUND;
  char *val = getenv("NOMP_INSTALL_DIR");
  if (val) {
    const char *python_dir = "python", *py_module = "c_to_loopy";
    size_t len0 = strlen(val), len1 = strlen(python_dir);

    char *scripts_path = (char *)calloc(len0 + len1 + 2, sizeof(char));
    strncpy(scripts_path, val, len0), strncpy(scripts_path + len0, "/", 1);
    strncpy(scripts_path + len0 + 1, python_dir, len1);

    append_to_sys_path(scripts_path);
    free(scripts_path);

    err = NOMP_C_TO_LOOPY_CONVERSION_ERROR;
    PyObject *pFile = PyUnicode_DecodeFSDefault(py_module),
             *pModule = PyImport_Import(pFile);
    Py_XDECREF(pFile);
    if (pModule) {
      PyObject *pFunc = PyObject_GetAttrString(pModule, py_module);
      if (pFunc && PyCallable_Check(pFunc)) {
        PyObject *pArgs = PyTuple_New(1), *pStr = PyUnicode_FromString(c_src);
        PyTuple_SetItem(pArgs, 0, pStr);
        pKnl = PyObject_CallObject(pFunc, pArgs);
        Py_XDECREF(pStr), Py_XDECREF(pArgs), Py_DECREF(pFunc), err = 0;
      }
      Py_DECREF(pModule);
    }
  }
  if (err)
    return err;

  // TODO: Apply Domain specific transformations

  // Call the user callback if present
  if (file && func) {
    err = NOMP_USER_CALLBACK_NOT_FOUND;
    PyObject *pFile = PyUnicode_DecodeFSDefault(file),
             *pModule = PyImport_Import(pFile), *pTransformedKnl = NULL;
    if (pModule) {
      PyObject *pFunc = PyObject_GetAttrString(pModule, func);
      if (pFunc && PyCallable_Check(pFunc)) {
        err = NOMP_USER_CALLBACK_FAILURE;
        PyObject *pArgs = PyTuple_New(1);
        PyTuple_SetItem(pArgs, 0, pKnl);
        pTransformedKnl = PyObject_CallObject(pFunc, pArgs);
        if (pTransformedKnl) {
          Py_DECREF(pKnl), err = 0;
          pKnl = pTransformedKnl;
        }
        Py_DECREF(pArgs), Py_DECREF(pFunc);
      }
      Py_DECREF(pModule);
    }
    Py_XDECREF(pFile);
  }
  if (err)
    return err;

  // Get grid size, OpenCL source, etc from transformed kernel
  if (pKnl) {
    // FIXME: This should only be done once
    PyObject *pLoopy = PyImport_ImportModule("loopy");
    if (pLoopy) {
      PyObject *pGenerateCodeV2 =
          PyObject_GetAttrString(pLoopy, "generate_code_v2");
      if (pGenerateCodeV2) {
        PyObject *pArgs = PyTuple_New(1);
        PyTuple_SetItem(pArgs, 0, pKnl);
        PyObject *pCode = PyObject_CallObject(pGenerateCodeV2, pArgs);
        PyObject *pDeviceCode =
            PyObject_GetAttrString(pCode, "device_code");
        PyObject *pDevCode = PyObject_CallObject(pDeviceCode, PyTuple_New(0));
        if (pCode) {
          PyObject_Print(pDevCode, stdout, Py_PRINT_RAW);
          Py_XDECREF(pCode);
        }
        Py_DECREF(pArgs);
        Py_DECREF(pGenerateCodeV2);
      }
      Py_DECREF(pLoopy);
    }
    Py_DECREF(pKnl);
  }

  return 0;
}
