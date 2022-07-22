#if !defined(_LIB_NOMP_IMPL_H_)
#define _LIB_NOMP_IMPL_H_

#include "nomp.h"
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct knl {
  char *src, *name;
  int ndim;
  size_t gsize[3], lsize[3];
};

struct prog {
  void *bptr;
};

struct mem {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
};

struct backend {
  int (*map)(struct backend *, struct mem *, const int);
  void (*map_ptr)(void **, size_t *, struct mem *);
  int (*knl_build)(struct backend *, struct prog *, const char *, const char *);
  int (*knl_set)(struct prog *, const int, const size_t, void *);
  int (*knl_run)(struct backend *, struct prog *, const int, const size_t *,
                 const size_t *);
  int (*knl_free)(struct prog *);
  int (*finalize)(struct backend *);
  void *bptr;
};

//==============================================================================
// OpenCL helper functions
//
int opencl_init(struct backend *ocl, const int platform_id,
                const int device_id);

//==============================================================================
// Python helper functions
//
int py_user_callback(struct knl *knl, const char *c_str, const char *file,
                     const char *func);

#endif // _LIB_NOMP_IMPL_H_
