#if !defined(_LIB_NOMP_IMPL_H_)
#define _LIB_NOMP_IMPL_H_

#define _POSIX_C_SOURCE 200809L

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUFSIZ 64
#define MAX_BACKEND_NAME_SIZE 32
#define MAX_SRC_SIZE 16384
#define MAX_CFLAGS_SIZE 16384

#define return_on_err(err)                                                     \
  {                                                                            \
    int err_ = (err);                                                          \
    if (nomp_get_log_type(err_) == NOMP_ERROR)                                 \
      return err_;                                                             \
  }

#include "nomp-aux.h"
#include "nomp-log.h"
#include "nomp-lpy.h"
#include "nomp-mem.h"

#include "nomp.h"

struct mem {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
};

/**
 * @ingroup nomp_mem_utils
 * @brief Returns the mem object corresponding to host pointer \p p.
 *
 * Returns the mem object corresponding to host ponter \p p. If no buffer has
 * been allocated for \p p on the device, returns NULL.
 *
 * @param[in] p Host pointer
 * @return struct mem *
 */
struct mem *mem_if_mapped(void *p);

struct prog {
  unsigned nargs, ndim;
  PyObject *py_global, *py_local, *py_dict;
  size_t global[3], local[3];
  void *bptr;
};

struct backend {
  char *backend, *install_dir, *annts_script, *annts_func;
  int platform_id, device_id, verbose;
  char name[MAX_BUFSIZ];
  int (*update)(struct backend *, struct mem *, const int);
  int (*knl_build)(struct backend *, struct prog *, const char *, const char *);
  int (*knl_run)(struct backend *, struct prog *, va_list);
  int (*knl_free)(struct prog *);
  int (*finalize)(struct backend *);
  void *bptr;
};

/**
 * @defgroup nomp_backend_utils Backend init functions
 */

/**
 * @ingroup nomp_backend_utils
 * @brief Initializes OpenCL backend with the specified platform and device.
 *
 * Initializes OpenCL backend while creating a command queue using the
 * given platform id and device id. Returns a negative value if an error
 * occured during the initialization, otherwise returns 0.
 *
 * @param[in] backend Target backend for code generation.
 * @param[in] platform_id Target platform id.
 * @param[in] device_id Target device id.
 * @return int
 */
int opencl_init(struct backend *backend, const int platform_id,
                const int device_id);

/**
 * @ingroup nomp_backend_utils
 * @brief Initializes Cuda backend with the specified platform and device.
 *
 * Initializes Cuda backend using the given device id. Platform id is not
 * used in the initialization of Cuda backend. Returns a negative value if an
 * error occured during the initialization, otherwise returns 0.
 *
 * @param[in] backend Target backend for code generation.
 * @param[in] platform_id Target platform id.
 * @param[in] device_id Target device id.
 * @return int
 */
int cuda_init(struct backend *backend, const int platform_id,
              const int device_id);

/**
 * @defgroup nomp_other_utils Other helper functions.
 */

/**
 * @ingroup nomp_compile_utils
 * @brief Compile a source string at runtime.
 *
 * Compile a source string at runtime using a specified compiler, flags and
 * a working directory. \p id is set to dynamically loaded \p entry point in the
 * compiled object file. \p id should be set to -1 on input and is set to a
 * non-negative value upon successful exit. On success, compile() returns 0 and
 * non-zero otherwise.
 *
 * @param[out] id Handle to the \p entry in the compiled binary file.
 * @param[in] source Source to be compiled at runtime.
 * @param[in] cc Full path to the compiler.
 * @param[in] cflags Compile flags to be used during compilation.
 * @param[in] entry Entry point (usually the name of function to be called) to
 * the source.
 * @param[in] wrkdir Working directory to generate outputs and store
 * temporaries.
 *
 * @return int
 */
int compile(int *id, const char *source, const char *cc, const char *cflags,
            const char *entry, const char *wrkdir);

#endif // _LIB_NOMP_IMPL_H_
