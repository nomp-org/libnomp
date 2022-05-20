#include <Python.h>

int import_loopy() {
  Py_Initialize();
  PyRun_SimpleString("from time import time");
}
