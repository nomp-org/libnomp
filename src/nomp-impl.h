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

#define MAX_BUFSIZ 128
#define MAX_BACKEND_SIZE 32
#define MAX_KEY_SIZE 128
#define MAX_SRC_SIZE 16384
#define MAX_CFLAGS_SIZE 16384
#define MAX_ARG_NAME_SIZE 128
#define MAX_FUNC_NAME_SIZE 128
#define MAX_KNL_ARGS 64

#include "nomp-aux.h"
#include "nomp-log.h"
#include "nomp-lpy.h"
#include "nomp-mem.h"

#include "nomp.h"

struct mem {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
  size_t bsize;
};

struct arg {
  char name[MAX_ARG_NAME_SIZE];
  size_t size;
  unsigned type;
  void *ptr;
};

struct prog {
  // Number of arguments of the kernel and meta info about
  // arguments.
  unsigned nargs;
  struct arg *args;
  // Dimension of kernel launch parameters, their pymbolic
  // expressions, and evaluated value of each dimension.
  unsigned ndim;
  PyObject *py_global, *py_local;
  size_t global[3], local[3];
  // Map of variable names and their values use to evaluate
  // the kernel launch parameters.
  PyObject *py_dict;
  // Pointer to keep track of backend specific data.
  void *bptr;
};

struct backend {
  // User configurations of the backend.
  int platform_id, device_id, verbose;
  char *backend, *install_dir;
  // Python function object which will be called to perform annotations.
  PyObject *py_annotate;
  // Pointers to backend functions used for backend dispatch.
  int (*update)(struct backend *, struct mem *, const int);
  int (*knl_build)(struct backend *, struct prog *, const char *, const char *);
  int (*knl_run)(struct backend *, struct prog *);
  int (*knl_free)(struct prog *);
  int (*sync)(struct backend *);
  int (*finalize)(struct backend *);
  // Pointer to keep track of backend specific data.
  void *bptr;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_backend_utils Backend init functions
 */

/**
 * @ingroup nomp_backend_utils
 * @brief Initializes OpenCL backend with the specified platform and device.
 *
 * Initializes OpenCL backend while creating a command queue using the
 * given platform id and device id. Returns a negative value if an error
 * occurred during the initialization, otherwise returns 0.
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
 * @brief Initializes SYCL backend with the specified platform and
 * device.
 *
 * Initializes SYCL backend while creating a command queue using the
 * given platform id and device id. Returns a positive value if an error
 * occurred during the initialization, otherwise returns 0.
 *
 * @param[in] backend Target backend for code generation.
 * @param[in] platform_id Target platform id.
 * @param[in] device_id Target device id.
 * @return int
 */
int sycl_init(struct backend *backend, const int platform_id,
              const int device_id);

/**
 * @ingroup nomp_backend_init
 * @brief Initializes Cuda backend with the specified platform and device.
 *
 * Initializes Cuda backend using the given device id. Platform id is not
 * used in the initialization of Cuda backend. Returns a negative value if an
 * error occurred during the initialization, otherwise returns 0.
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
 * error occurred during the initialization, otherwise returns 0.
 *
 * @param[in] backend Target backend for code generation.
 * @param[in] platform_id Target platform id.
 * @param[in] device_id Target device id.
 * @return int
 */
int hip_init(struct backend *backend, const int platform_id,
             const int device_id);

/**
 * @ingroup nomp_backend_init
 * @brief Initializes ISPC backend with the specified platform and device.
 *
 * Initializes ISPC backend using the given device id and platform type.
 * Returns a negative value if an error occurred during the initialization,
 * otherwise returns 0.
 *
 * @param[in] backend Target backend for code generation.
 * @param[in] platform_type Target platform type.
 * @param[in] device_id Target device id.
 * @return int
 */
int ispc_init(struct backend *backend, const int platform_type,
              const int device_id);

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_IMPL_H_
