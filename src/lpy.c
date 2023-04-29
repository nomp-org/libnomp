#include "nomp-impl.h"
#include <symengine/cwrapper.h>

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

int nomp_py_realize_reduction(PyObject **knl, const char *var) {
  int err = 1;
  PyObject *reduction = PyUnicode_FromString(module_reduction);
  if (reduction) {
    PyObject *module = PyImport_Import(reduction);
    if (module) {
      PyObject *rr = PyObject_GetAttrString(module, realize_reduction);
      if (rr) {
        PyObject *pvar = PyUnicode_FromString(var);
        PyObject *result = PyObject_CallFunctionObjArgs(rr, *knl, pvar, NULL);
        if (result)
          Py_DECREF(*knl), *knl = result, err = 0;
        Py_XDECREF(pvar), Py_DECREF(rr);
      }
      Py_DECREF(module);
    }
    Py_DECREF(reduction);
  }
  if (err) {
    return nomp_set_log(NOMP_PY_CALL_FAILURE, NOMP_ERROR,
                        "Call to realize_reduction() failed.");
  }

  return 0;
}

int nomp_py_c_to_loopy(PyObject **knl, const char *src, const char *backend) {
  int err = 1;
  PyObject *lpy_api = PyUnicode_FromString(module_loopy_api);
  if (lpy_api) {
    PyObject *module = PyImport_Import(lpy_api);
    if (module) {
      PyObject *c_to_lpy = PyObject_GetAttrString(module, c_to_loopy);
      if (c_to_lpy) {
        PyObject *psrc = PyUnicode_FromString(src);
        PyObject *pbackend = PyUnicode_FromString(backend);
        *knl = PyObject_CallFunctionObjArgs(c_to_lpy, psrc, pbackend, NULL);
        err = (*knl == NULL);
        Py_XDECREF(psrc), Py_XDECREF(pbackend), Py_DECREF(c_to_lpy);
      }
      Py_DECREF(module);
    }
    Py_DECREF(lpy_api);
  }
  if (err) {
    return nomp_set_log(NOMP_LOOPY_CONVERSION_FAILURE, NOMP_ERROR,
                        "C to Loopy conversion failed.\n");
  }

  return 0;
}

int nomp_py_set_annotate_func(PyObject **annotate_func, const char *path_) {
  // Find file and function from path.
  char *path = strndup(path_, PATH_MAX + NOMP_MAX_BUFSIZ);
  char *file = strtok(path, "::"), *func = strtok(NULL, "::");
  if (file == NULL || func == NULL) {
    nomp_free(&path);
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

  nomp_free(&path);
  return err;
}

int nomp_py_apply_annotations(PyObject **knl, PyObject *func,
                              const PyObject *annts, const PyObject *context) {
  if (knl && *knl && func) {
    if (PyCallable_Check(func)) {
      PyObject *tknl =
          PyObject_CallFunctionObjArgs(func, *knl, annts, context, NULL);
      if (tknl)
        Py_DECREF(*knl), *knl = tknl;
    }
  }

  return 0;
}

int nomp_py_apply_transform(PyObject **knl, const char *file, const char *func,
                            const PyObject *context) {
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
        PyObject *tknl =
            PyObject_CallFunctionObjArgs(pfunc, *knl, context, NULL);
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

char *str_replace(const char *orig, char *rep, char *with) {
  char *result, *tmp;
  int len_rep, len_with, len_front, count = 0;

  if (!orig || !rep)
    return NULL;
  len_rep = strnlen(rep, MAX_BUFSIZ);
  if (len_rep == 0)
    return NULL;
  if (!with)
    with = "";
  len_with = strnlen(with, MAX_BUFSIZ);

  char *ins = (char *)malloc((strnlen(orig, MAX_BUFSIZ) + 1) * sizeof(char));
  strncpy(ins, orig, strnlen(orig, MAX_BUFSIZ) + 1);
  while ((tmp = strstr(ins, rep)) != NULL) {
    count++;
    ins = tmp + len_rep;
  }

  tmp = result =
      malloc(strnlen(orig, MAX_BUFSIZ) + (len_with - len_rep) * count + 1);

  if (!result)
    return NULL;

  while (count--) {
    ins = strstr(orig, rep);
    len_front = ins - orig;
    tmp = strncpy(tmp, orig, len_front) + len_front;
    tmp = strncpy(tmp, with, len_with) + len_with;
    orig += len_front + len_rep;
  }
  strncpy(tmp, orig, strnlen(orig, MAX_BUFSIZ) + 1);
  return result;
}

int sym_c_vec_push(CVecBasic *vec, const char *str) {
  CWRAPPER_OUTPUT_TYPE err;
  basic a;
  basic_new_stack(a);
  err = basic_parse(a, str);
  vecbasic_push_back(vec, a);
  basic_free_stack(a);
  return err;
}

int sym_c_map_push(CMapBasicBasic *map, const char *key, const char *val) {
  basic a, b;
  basic_new_stack(a);
  basic_new_stack(b);
  symbol_set(a, key);
  integer_set_str(b, val);
  mapbasicbasic_insert(map, a, b);
  basic_free_stack(a);
  basic_free_stack(b);
  return 0;
}

int sym_evaluate(size_t *out, unsigned i, CVecBasic *vec, CMapBasicBasic *map) {
  CWRAPPER_OUTPUT_TYPE err;
  basic a, b, c;
  basic_new_stack(a);
  basic_new_stack(b);
  basic_new_stack(c);
  vecbasic_get(vec, i, b);
  basic_subs(a, b, map);
  err = basic_evalf(c, a, 53, 1);
  basic_floor(c, c);
  out[i] = integer_get_ui(c);
  basic_free_stack(a);
  basic_free_stack(b);
  basic_free_stack(c);
  return err;
}

int py_get_grid_size_aux(PyObject *exp, CVecBasic *vec) {
  int err = 1;
  PyObject *strifier = PyObject_CallMethod(exp, make_stringifier, NULL);
  if (strifier) {
    PyObject *str_op = PyObject_CallFunctionObjArgs(strifier, exp, NULL);
    if (str_op) {
      const char *str = PyUnicode_AsUTF8(str_op);
      char *s_rep = str_replace(str, "//", "/");
      err = sym_c_vec_push(vec, s_rep);
      Py_DECREF(str_op);
    }
    Py_DECREF(strifier);
  } else {
    PyObject *obj_rep = PyObject_Repr(exp);
    const char *str = PyUnicode_AsUTF8(obj_rep);
    err = sym_c_vec_push(vec, str);
    Py_XDECREF(obj_rep);
  }
  if (err) {
    return nomp_set_log(
        NOMP_SYMENGINE_PARSING_FAILURE, NOMP_ERROR,
        "Unable to parse and and assign value using SymEngine.");
  }
  return err;
}

int py_get_grid_size(struct prog *prg, PyObject *knl) {
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
          Py_DECREF(expr);
          if (grid_size && PyTuple_Check(grid_size)) {
            PyObject *py_global = PyTuple_GetItem(grid_size, 0);
            PyObject *py_local = PyTuple_GetItem(grid_size, 1);
            prg->sym_global = vecbasic_new();
            prg->sym_local = vecbasic_new();
            prg->ndim = PyTuple_Size(py_global);

            for (int i = 0; i < PyTuple_Size(py_global); i++) {
              PyObject *ele = PyTuple_GetItem(py_global, i);
              py_get_grid_size_aux(ele, prg->sym_global);
            }

            for (int i = 0; i < PyTuple_Size(py_local); i++) {
              PyObject *ele = PyTuple_GetItem(py_local, i);
              py_get_grid_size_aux(ele, prg->sym_local);
            }

            if (PyTuple_Size(py_local) > prg->ndim)
              prg->ndim = PyTuple_Size(py_local);
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

int py_eval_grid_size(struct prog *prg) {
  // If the expressions are not NULL, iterate through them and evaluate with
  // pymbolic. Also, we should calculate and store a hash of the dict that
  // is passed. If the hash is the same, no need of re-evaluating the grid
  // size.
  for (unsigned i = 0; i < 3; i++)
    prg->global[i] = prg->local[i] = 1;

  for (unsigned j = 0; j < vecbasic_size(prg->sym_global); j++) {
    nomp_check(sym_evaluate(prg->global, j, prg->sym_global, prg->map));
  }

  for (unsigned j = 0; j < vecbasic_size(prg->sym_local); j++) {
    nomp_check(sym_evaluate(prg->local, j, prg->sym_local, prg->map));
  }

  return 0;
}
