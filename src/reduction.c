#include "nomp-reduction.h"

#define NOMP_DO_SUM(a, b) (a) += (b)
#define NOMP_DO_PROD(a, b) (a) *= (b)

#define NOMP_INIT_SUM(a) (a) = 0
#define NOMP_INIT_PROD(a) (a) = 1

#define NOMP_FOR_EACH_OP(T, macro) macro(T, SUM) macro(T, PROD)
#define NOMP_FOR_EACH_DOMAIN(macro) macro(int) macro(float)

#define NOMP_REDUCTION(T, OP)                                                  \
  static void reduce_##T##_##OP(T *out, T *in, unsigned n) {                   \
    NOMP_INIT_##OP(*out);                                                      \
    for (; n; n--)                                                             \
      NOMP_DO_##OP(*out, in[n - 1]);                                           \
  }

#define DEFINE_FOR_OP(T) NOMP_FOR_EACH_OP(T, NOMP_REDUCTION)
#define DEFINE_REDUCTION NOMP_FOR_EACH_DOMAIN(DEFINE_FOR_OP)
DEFINE_REDUCTION
#undef DEFINE_REDUCTION
#undef DEFINE_FOR_OP

// FIXME: Temporary hack -- should be fixed
#define NOMP_int NOMP_INT
#define NOMP_float NOMP_FLOAT

#define SWITCH_OP_CASE(T, OP)                                                  \
  case NOMP_##OP:                                                              \
    WITH_OP(T, OP);                                                            \
    break;

#define NOMP_SWITCH_OP(T, OP)                                                  \
  switch (op) {                                                                \
    NOMP_FOR_EACH_OP(T, SWITCH_OP_CASE)                                        \
  default:                                                                     \
    break;                                                                     \
  }

#define SWITCH_DOMAIN_CASE(T)                                                  \
  case NOMP_##T:                                                               \
    WITH_DOMAIN(T);                                                            \
    break;

#define NOMP_SWITCH_DOMAIN(dom)                                                \
  {                                                                            \
    switch (dom) {                                                             \
      NOMP_FOR_EACH_DOMAIN(SWITCH_DOMAIN_CASE)                                 \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  }

int host_side_reduction(struct backend *backend, struct prog *prg,
                        struct mem *m) {
  nomp_check(backend->update(backend, m, NOMP_FROM));

  struct arg *args = prg->args;
  // int index = prg->reduction_indx, op = prg->reduction_op;
  int dom, op;
  void *ptr;

#define WITH_OP(T, OP) reduce_##T##_##OP((T *)ptr, (T *)m->hptr, prg->global[0])
#define WITH_DOMAIN(T) NOMP_SWITCH_OP(T, OP)
  NOMP_SWITCH_DOMAIN(dom);
#undef WITH_DOMAIN
#undef WITH_OP

  return 0;
}

int py_handle_reduction(PyObject **knl, int *operator, const char * backend) {
  int err = 1;
  PyObject *reduction = PyUnicode_FromString("reduction");
  PyObject *ret = NULL, *op = NULL;
  if (reduction) {
    PyObject *module = PyImport_Import(reduction);
    Py_DECREF(reduction);
    if (module) {
      PyObject *function = PyObject_GetAttrString(module, "realize_reduction");
      Py_DECREF(module);
      if (function && PyCallable_Check(function)) {
        PyObject *bnd = PyUnicode_FromString(backend);
        PyObject *ret = PyObject_CallFunctionObjArgs(function, *knl, bnd, NULL);
        Py_DECREF(function);
        if (PyTuple_Check(ret) && PyTuple_GET_SIZE(ret) == 2) {
          Py_DECREF(*knl), *knl = PyTuple_GetItem(ret, 0);
          PyObject *op = PyTuple_GetItem(ret, 1);
          *operator= PyLong_AsLong(op), Py_DECREF(op), err = 0;
        }
        Py_XDECREF(bnd);
      }
    }
  }
  if (err) {
    return nomp_set_log(NOMP_PY_CALL_FAILURE, NOMP_ERROR,
                        "Calling realize_reduction failed.");
  }
  return 0;
}
