#if !defined(_LIB_NOMP_IMPL_H_)
#define _LIB_NOMP_IMPL_H_

#define _POSIX_C_SOURCE 200809L

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <symengine/cwrapper.h>

#include <assert.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NOMP_MAX_BUFSIZ 128
#define NOMP_MAX_SRC_SIZE 16384
#define NOMP_MAX_CFLAGS_SIZE 16384
#define NOMP_MAX_KNL_ARGS 64
#define NOMP_MAX_SCRATCH_SIZE (1024 * sizeof(double))
#define NOMP_MEM_BPTR_OFFSET(m, start) (((m)->usize) * ((start) - (m)->idx0))
#define NOMP_MEM_HPTR_OFFSET(m, start) ((m)->usize * (start))
#define NOMP_MEM_BYTES(m, start, end) (((end) - (start)) * ((m)->usize))

#include "nomp-aux.h"
#include "nomp-log.h"
#include "nomp-lpy.h"
#include "nomp-mem.h"

#include "nomp.h"

struct nomp_mem {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
  size_t bsize;
};

struct nomp_arg {
  char name[NOMP_MAX_BUFSIZ];
  size_t size;
  unsigned type;
  void *ptr;
};

struct nomp_prog {
  // Number of arguments of the kernel and meta info about
  // arguments.
  unsigned nargs;
  struct nomp_arg *args;
  // Dimension of kernel launch parameters, their pymbolic
  // expressions, and evaluated value of each dimension.
  unsigned ndim;
  CVecBasic *sym_global, *sym_local;
  size_t global[3], local[3], gws[3];
  // Map of variable names and their values use to evaluate
  // the kernel launch parameters.
  CMapBasicBasic *map;
  // Boolean flag to determine if the grid size should be evaluated or not.
  int is_grid_eval;
  // Pointer to keep track of backend specific data.
  void *bptr;
  // Reduction related metadata.
  int reduction_index, reduction_op, reduction_type, reduction_size;
  void *reduction_ptr;
};

struct nomp_backend {
  // User configurations of the backend.
  int platform_id, device_id, verbose, profile;
  char backend[NOMP_MAX_BUFSIZ], install_dir[PATH_MAX];
  // Pointers to backend functions used for backend dispatch.
  int (*update)(struct nomp_backend *, struct nomp_mem *,
                const nomp_map_direction_t op, size_t start, size_t end);
  int (*knl_build)(struct nomp_backend *, struct nomp_prog *, const char *,
                   const char *);
  int (*knl_run)(struct nomp_backend *, struct nomp_prog *);
  int (*knl_free)(struct nomp_prog *);
  int (*sync)(struct nomp_backend *);
  int (*finalize)(struct nomp_backend *);
  // Scratch memory to be used as temporary memory for kernels (like
  // reductions).
  struct nomp_mem scratch;
  // Python function object which will be called to perform annotations.
  PyObject *py_annotate;
  // Context info is used to pass necessary infomation to kernel
  // transformations and annotations.
  PyObject *py_context;
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
int opencl_init(struct nomp_backend *backend, const int platform_id,
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
int sycl_init(struct nomp_backend *backend, const int platform_id,
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
int cuda_init(struct nomp_backend *backend, const int platform_id,
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
int hip_init(struct nomp_backend *backend, const int platform_id,
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
int ispc_init(struct nomp_backend *backend, const int platform_type,
              const int device_id);

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_IMPL_H_
