#if !defined(_LIB_NOMP_IMPL_H_)
#define _LIB_NOMP_IMPL_H_

#include <nomp.h>

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int opencl_init(struct backend *ocl, const int platform_id,
                const int device_id);

#endif // _LIB_NOMP_IMPL_H_
