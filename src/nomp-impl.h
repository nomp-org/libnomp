#if !defined(_LIB_NOMP_IMPL_H_)
#define _LIB_NOMP_IMPL_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nomp.h"

#define py_dir "python"
#define py_module "loopy_api"
#define py_func "c_to_loopy"

struct prog {
  void *bptr;
};

struct mem {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
};

struct error {
  const char *description;
  const char *file_name;
  unsigned line_no;
};

struct error_stack {
  struct error *stack;
  int next_error_id;
};

struct backend {
  char name[BUFSIZ];
  int (*map)(struct backend *, struct mem *, const int);
  int (*knl_build)(struct backend *, struct prog *, const char *, const char *);
  int (*knl_run)(struct backend *, struct prog *, const int, const size_t *,
                 const size_t *, int, va_list);
  int (*knl_free)(struct prog *);
  int (*finalize)(struct backend *);
  void *bptr;
};

struct mem *mem_if_mapped(void *p);

//==============================================================================
// Backend init functions
//
int opencl_init(struct backend *backend, const int platform_id,
                const int device_id);
int cuda_init(struct backend *backend, const int platform_id,
              const int device_id);

//==============================================================================
// Python helper functions
//
int py_append_to_sys_path(const char *path);
int py_c_to_loopy(PyObject **pKnl, const char *c_src, const char *backend);
int py_user_callback(PyObject **pKnl, const char *file, const char *func);
int py_get_knl_name_and_src(char **name, char **src, PyObject *pKnl);
int py_get_grid_size(int *ndim, size_t *global, size_t *local, PyObject *pKnl,
                     PyObject *pDict);

//==============================================================================
// Other helper functions
//
char *strcatn(int nstr, ...);

#endif // _LIB_NOMP_IMPL_H_