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

int py_user_callback(struct knl *knl, const char *c_src, const char *file) {
  if (!Py_IsInitialized()) {
    Py_Initialize();
    // FIXME: Ugly and check for error
    PyRun_SimpleString("import sys\nsys.path.append(\".\")");
  }

  // TODO: Create a loopy kernel from the `knl` string
  PyObject *pKnl = NULL;

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
  // FIXME: Should I free knlName?
  const char *knlName = PyUnicode_AsUTF8(pKnlName);

  // Call the user callback
  PyObject *pFile = PyUnicode_DecodeFSDefault(file), *pTransformedKnl = NULL;
  PyObject *pModule = PyImport_Import(pFile);
  Py_XDECREF(pFile);
  if (pModule) {
    PyObject *pFunc = PyObject_GetAttrString(pModule, knlName);
    if (pFunc && PyCallable_Check(pFunc)) {
      PyObject *pArgs = PyTuple_New(1);
      PyTuple_SetItem(pArgs, 0, pKnl);
      pTransformedKnl = PyObject_CallObject(pFunc, pArgs);
      Py_DECREF(pArgs);
      Py_DECREF(pFunc);
    }
    Py_DECREF(pModule);
  }

  // Get grid size, OpenCL source, etc from transformed kernel
  if (pTransformedKnl) {
    PyObject *pLoopy = PyImport_ImportModule("loopy");
    if (pLoopy) {
      PyObject *pGenerateCodeV2 =
          PyObject_GetAttrString(pLoopy, "generate_code_v2");
      if (pGenerateCodeV2) {
        PyObject *pArgs = PyTuple_New(1);
        PyTuple_SetItem(pArgs, 0, pTransformedKnl);
        PyObject *pCode = PyObject_CallObject(pGenerateCodeV2, pArgs);
        PyObject_Print(pCode, stdout, Py_PRINT_RAW);
        Py_XDECREF(pCode);
        Py_DECREF(pArgs);
        Py_DECREF(pGenerateCodeV2);
      }
      Py_DECREF(pLoopy);
    }
    Py_DECREF(pTransformedKnl);
  }

  Py_XDECREF(pKnlName), Py_XDECREF(pKnl);

  return 0;
}
