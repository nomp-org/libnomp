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
#define NOMP_MAX_SCRATCH_SIZE (32768 * sizeof(double))

#include "nomp-aux.h"
#include "nomp-log.h"
#include "nomp-loopy.h"
#include "nomp-mem.h"

#include "nomp.h"

/**
 * @defgroup nomp_structs C Structs
 * @brief C structs used in libnomp
 */

/**
 * @ingroup nomp_structs
 *
 * @brief Represents a memory block that can be used for data storage and
 * transfer between a host and a device.
 */
struct nomp_mem_t {
  /**
   * @brief Starting index of a memory block.
   */
  size_t idx0;
  /**
   * @brief Ending index of a memory block.
   */
  size_t idx1;
  /**
   * @brief Size (in bytes) of each element in the memory block.
   */
  size_t usize;
  /**
   * @brief Pointer to the host memory.
   */
  void *hptr;
  /**
   * @brief Pointer to the device memory.
   */
  void *bptr;
  /**
   * @brief Size (in bytes) of allocated memory on device.
   */
  size_t bsize;
};

#define NOMP_MEM_OFFSET(start, usize) ((start) * (usize))
#define NOMP_MEM_BYTES(start, end, usize) (((end) - (start)) * (usize))

/**
 * @ingroup nomp_structs
 *
 * @brief Represents a kernel argument.
 */
struct nomp_arg_t {
  /**
   * @brief Size (in bytes) of allocated memory on device.
   */
  char name[NOMP_MAX_BUFSIZ];
  /**
   * @brief The name of the argument.
   */
  size_t size;
  /**
   * @brief The size of the argument data.
   */
  unsigned type;
  /**
   * @brief A pointer to the argument data.
   */
  void *ptr;
};

/**
 * @ingroup nomp_structs
 *
 * @brief Struct to store meta information about kernel arguments.
 */
struct nomp_prog_t {
  /**
   * @brief Number of kernel arguments.
   */
  unsigned nargs;
  /**
   * @brief Pointer to an array of kernel arguments.
   */
  struct nomp_arg_t *args;
  /**
   * @brief Dimension of kernel launch parameters.
   */
  unsigned ndim;
  /**
   * @brief Pymbolic expressions for global dimensions.
   */
  CVecBasic *sym_global;
  /**
   * @brief Pymbolic expressions for local dimensions.
   */
  CVecBasic *sym_local;
  /**
   * @brief Sizes of each global dimensions.
   */
  size_t global[3];
  /**
   * @brief Sizes of each local dimensions.
   */
  size_t local[3];
  /**
   * @brief Global work size.
   */
  size_t gws[3];
  /**
   * @brief Map of variable names and their values use to evaluate the kernel
   * launch parameters.
   */
  CMapBasicBasic *map;
  /**
   * @brief Boolean flag to determine if the grid size should be evaluated or
   * not.
   */
  int eval_grid;
  /**
   * @brief Pointer to backend specific data.
   */
  void *bptr;
  /**
   * @brief Reduction kernel id.
   */
  int redn_idx;
  /**
   * @brief Reduction operator.
   */
  int redn_op;
  /**
   * @brief Type of reduction.
   */
  int redn_type;
  /**
   * @brief Size of the array to be reduced.
   */
  int redn_size;
  /**
   * @brief A pointer to data to be reduced.
   */
  void *redn_ptr;
};

/**
 * @ingroup nomp_structs
 *
 * @brief Struct to store user configurations and pointers to backend functions.
 */
struct nomp_backend_t {
  // User configurations of the backend.
  /**
   * @brief Platform ID of the backend.
   */
  int platform_id;
  /**
   * @brief Device ID of the backend.
   */
  int device_id;
  /**
   * @brief Verbosity level.
   */
  int verbose;
  /**
   * @brief Profiler level.
   */
  int profile;
  /**
   * @brief Name of the backend.
   */
  char backend[NOMP_MAX_BUFSIZ];
  /**
   * @brief Nomp installation directory.
   */
  char install_dir[PATH_MAX];
  // Pointers to backend functions used for backend dispatch.
  /**
   * @brief Pointer to backend memory update function.
   */
  int (*update)(struct nomp_backend_t *, struct nomp_mem_t *,
                const nomp_map_direction_t op, size_t start, size_t end,
                size_t usize);
  /**
   * @brief Pointer to backend kernel build function.
   */
  int (*knl_build)(struct nomp_backend_t *, struct nomp_prog_t *, const char *,
                   const char *);
  /**
   * @brief Pointer to backend kernel run function.
   */
  int (*knl_run)(struct nomp_backend_t *, struct nomp_prog_t *);
  /**
   * @brief Pointer to backend kernel free function.
   */
  int (*knl_free)(struct nomp_prog_t *);
  /**
   * @brief Pointer to backend synchronization function.
   */
  int (*sync)(struct nomp_backend_t *);
  /**
   * @brief Pointer to backend finalizing function.
   */
  int (*finalize)(struct nomp_backend_t *);
  /**
   * @brief Scratch memory to be used as temporary memory for kernels (like
   * reductions).
   */
  struct nomp_mem_t scratch;
  /**
   * @brief Python function object which will be called to perform annotations.
   */
  PyObject *py_annotate;
  /**
   * @brief Context info is used to pass necessary infomation to kernel
   * transformations and annotations.
   */
  PyObject *py_context;
  /**
   * @brief Pointer to backend specific data.
   */
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
int opencl_init(struct nomp_backend_t *backend, const int platform_id,
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
int sycl_init(struct nomp_backend_t *backend, const int platform_id,
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
int cuda_init(struct nomp_backend_t *backend, const int platform_id,
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
int hip_init(struct nomp_backend_t *backend, const int platform_id,
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
int ispc_init(struct nomp_backend_t *backend, const int platform_type,
              const int device_id);

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_IMPL_H_
