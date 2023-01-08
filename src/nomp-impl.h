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
#define MAX_BACKEND_SIZE 32
#define MAX_SRC_SIZE 16384
#define MAX_CFLAGS_SIZE 16384

#ifdef __cplusplus
extern "C" {
#endif

#include "nomp-aux.h"
#include "nomp-log.h"
#include "nomp-lpy.h"
#include "nomp-mem.h"

#include "nomp.h"

struct mem {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
};

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
  int (*sync)(struct backend *);
  int (*finalize)(struct backend *);
  void *bptr;
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
 * @ingroup nomp_backend_init
 * @brief Initializes SYCL Opencl backend with the specified platform and
 * device.
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
int sycl_opencl_init(struct backend *backend, const int platform_id,
                     const int device_id);

/**
 * @ingroup nomp_backend_init
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
 * @ingroup nomp_backend_init
 * @brief Initializes HIP backend with the specified platform and device.
 *
 * Initializes HIP backend using the given device id. Platform id is not
 * used in the initialization of HIP backend. Returns a negative value if an
 * error occured during the initialization, otherwise returns 0.
 *
 * @param[in] backend Target backend for code generation.
 * @param[in] platform_id Target platform id.
 * @param[in] device_id Target device id.
 * @return int
 */
int hip_init(struct backend *backend, const int platform_id,
             const int device_id);

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_IMPL_H_

#ifdef __cplusplus
}
#endif
