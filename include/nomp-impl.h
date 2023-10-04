#if !defined(_LIB_NOMP_IMPL_H_)
#define _LIB_NOMP_IMPL_H_

#define _POSIX_C_SOURCE 200809L

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <symengine/cwrapper.h>

#include "nomp-defs.h"
#include "nomp-log.h"
#include "nomp-mem.h"
#include "nomp.h"

/**
 * @defgroup nomp_internal_types Internal types
 * @brief Internal types used in libnomp.
 */

/**
 * @ingroup nomp_internal_types
 *
 * @brief Structure to keep track of nomp runtime configuration.
 */
typedef struct {
  /**
   * ID of the platform to be used for the backend. This is only used when the
   * backend is OpenCL.
   */
  int platform;
  /**
   * ID of the device to be used with the backend.
   */
  int device;
  /**
   * Verbose level to be used for displaying log messages.
   */
  int verbose;
  /**
   * Turn profiling on or off.
   */
  int profile;
  /**
   * Name of the backend to be used (OpenCL, CUDA, etc.).
   */
  char backend[NOMP_MAX_BUFFER_SIZE + 1];
  /**
   * Installed directory of the library.
   */
  char install_dir[PATH_MAX + 1];
  /**
   * Directory where transform and annotations scripts are located.
   */
  char scripts_dir[PATH_MAX + 1];
} nomp_config_t;

/**
 * @ingroup nomp_internal_types
 *
 * @brief Structure to keep track of arguments (variables) of a nomp program
 * (::nomp_prog_t).
 */
typedef struct {
  /**
   * Name of the argument (respective variable name in C).
   */
  char name[NOMP_MAX_BUFFER_SIZE + 1];
  /**
   * Size of the argument given by sizeof() operator.
   */
  size_t size;
  /**
   * Type of the argument (one of ::nomp_type_t).
   */
  nomp_type_t type;
  /**
   * Pointer to the argument.
   */
  void *ptr;
} nomp_arg_t;

/**
 * @ingroup nomp_internal_types
 *
 * @brief Structure to keep track of nomp program.
 */
typedef struct {
  /**
   * Number of arguments of the kernel.
   */
  unsigned nargs;
  /**
   * Pointer to the arguments of the kernel.
   */
  nomp_arg_t *args;
  /**
   * Dimension of kernel launch parameters.
   */
  unsigned ndim;
  /**
   * SymEngine expressions of kernel launch parameters as a vector.
   */
  CVecBasic *sym_global, *sym_local;
  /**
   * Map of variable names and their values use to evaluate
   * the kernel launch parameters.
   */
  CMapBasicBasic *map;
  /**
   * Evaluated value of kernel launch parameters (re-evaluated if the
   * input changes).
   */
  size_t global[3], local[3], gws[3];
  // Boolean flag to determine if the grid size should be evaluated or not.
  int eval_grid;
  // Pointer to keep track of backend specific data.
  void *bptr;
  // Reduction related metadata.
  int redn_idx, redn_op, redn_type, redn_size;
  void *redn_ptr;
} nomp_prog_t;

typedef struct {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
  size_t bsize;
} nomp_mem_t;

/**
 * @ingroup nomp_internal_types
 *
 * @brief Structure to keep track of nomp runtime data and backend specific
 * data. This structure is also used to dispatch backend specific functions.
 */
struct nomp_backend {
  /**
   * Function pointer to the backend update function which can allocate,
   * update and free backend memory.
   */
  int (*update)(struct nomp_backend *, nomp_mem_t *,
                const nomp_map_direction_t op, size_t start, size_t end,
                size_t usize);
  /**
   * Function pointer to the backend kernel build function.
   */
  int (*knl_build)(struct nomp_backend *, nomp_prog_t *, const char *,
                   const char *);
  /**
   * Function pointer to the backend kernel run function.
   */
  int (*knl_run)(struct nomp_backend *, nomp_prog_t *);
  /**
   * Function pointer to the backend kernel free function.
   */
  int (*knl_free)(nomp_prog_t *);
  /**
   * Function pointer to the backend synchronization function.
   */
  int (*sync)(struct nomp_backend *);
  /**
   * Function pointer to the backend finalize function which releases allocated
   * resources.
   */
  int (*finalize)(struct nomp_backend *);

  /**
   * Scratch memory to be used as temporary memory for kernels (like reductions)
   */
  nomp_mem_t scratch;

  /**
   * Python function object which will be called to perform annotations.
   */
  PyObject *py_annotate;

  /**
   * Context info is used to pass necessary information to kernel
   * transformations and annotations.
   */
  PyObject *py_context;

  /**
   * Pointer to keep track of backend specific data. This is allocated and
   * released by the backend.
   */
  void *bptr;
};

typedef struct nomp_backend nomp_backend_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_internal_macros Internal macros
 * @brief Internal macros used in libnomp.
 */

/**
 * @ingroup nomp_internal_macros
 *
 * @def nomp_check
 *
 * @brief Check if nomp API return value is an error. In case of an error,
 * non-zero error code is returned to the user. Otherwise, the return value is
 * zero.
 *
 * @param[in] err Return value from nomp API.
 *
 */
#define nomp_check(err)                                                        \
  {                                                                            \
    int err_ = (err);                                                          \
    if (err_ > 0)                                                              \
      return err_;                                                             \
  }

#define NOMP_MEM_OFFSET(start, usize) ((start) * (usize))
#define NOMP_MEM_BYTES(start, end, usize) (((end) - (start)) * (usize))

/**
 * @defgroup nomp_backend_init Backend initialization functions
 *
 * @brief Functions for initializing different backend as OpenCL, CUDA, etc.
 * These has to be refactored to handle the case when the backend is not
 * available.
 */

int opencl_init(nomp_backend_t *backend, int platform, int device);

int cuda_init(nomp_backend_t *backend, int platform, int device);

int hip_init(nomp_backend_t *backend, int platform, int device);

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_IMPL_H_
