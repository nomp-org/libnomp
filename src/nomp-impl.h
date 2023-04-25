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
#define MAX_VAL_SIZE 128
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

/**
 * @brief Represents a memory block that can be used for data storage and
 * transfer between a host and a device.
 */
struct mem {
  size_t idx0;  /**< Starting index of a memory block */
  size_t idx1;  /**< Ending index of a memory block */
  size_t usize; /**< Size (in bytes) of each element in the memory block */
  void *hptr;   /**< Pointer to the host memory */
  void *bptr;   /**< Pointer to the device memory */
  size_t bsize; /**< Size (in bytes) of allocated memory on device */
};

/**
 * @brief Represents a kernel argument.
 */
struct arg {
  char name[MAX_ARG_NAME_SIZE]; /**< The name of the argument */
  size_t size;                  /**< The size of the argument data */
  unsigned type;                /**< The type of the argument */
  void *ptr;                    /**< A pointer to the argument data */
};

/**
 * @brief Struct to store meta information about kernel arguments.
 */
struct prog {
  unsigned nargs;      /**< Number of kernel arguments */
  struct arg *args;    /**< Pointer to an array of kernel arguments */
  unsigned ndim;       /**< Dimension of kernel launch parameters */
  PyObject *py_global; /**< Pymbolic expressions for global dimensions */
  PyObject *py_local;  /**< Pymbolic expressions for local dimensions */
  size_t global[3];    /**< Sizes of each global dimension */
  size_t local[3];     /**< Sizes of each local dimension */
  PyObject *py_dict; /**< Map of variable names to their values used to evaluate
                        the kernel launch parameters */
  void *bptr;        /**< Pointer to backend specific data */
};

/**
 * @brief Struct to store user configurations and pointers to backend functions.
 */
struct backend {
  int platform_id;       /**< Platform ID of the backend */
  int device_id;         /**< Device ID of the backend */
  int verbose;           /**< Verbosity level */
  char *backend;         /**< Name of the backend */
  char *install_dir;     /**< Nomp installation directory */
  PyObject *py_annotate; /**< Python function object which will be called to
                            perform annotations */
  int (*update)(struct backend *, struct mem *,
                const int); /**< Pointer to backend memory update function */
  int (*knl_build)(
      struct backend *, struct prog *, const char *,
      const char *); /**< Pointer to backend kernel build function */
  int (*knl_run)(struct backend *,
                 struct prog *); /**< Pointer to backend kernel run function */
  int (*knl_free)(
      struct prog *); /**< Pointer to backend kernel free function */
  int (*sync)(
      struct backend *); /**< Pointer to backend synchronization function */
  int (*finalize)(
      struct backend *); /**< Pointer to backend finalizing function */
  void *bptr;            /**< Pointer to backend specific data */
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
