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

#include "nomp-aux.h"
#include "nomp-defs.h"
#include "nomp-log.h"
#include "nomp-loopy.h"
#include "nomp-mem.h"
#include "nomp.h"

struct nomp_mem_t {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
  size_t bsize;
};

#define NOMP_MEM_OFFSET(start, usize) ((start) * (usize))
#define NOMP_MEM_BYTES(start, end, usize) (((end) - (start)) * (usize))

struct nomp_arg_t {
  char name[NOMP_MAX_BUFFER_SIZE + 1];
  size_t size;
  unsigned type;
  void *ptr;
};

struct nomp_prog_t {
  // Number of arguments of the kernel and meta info about
  // arguments.
  unsigned nargs;
  struct nomp_arg_t *args;
  // Dimension of kernel launch parameters, their pymbolic
  // expressions, and evaluated value of each dimension.
  unsigned ndim;
  CVecBasic *sym_global, *sym_local;
  size_t global[3], local[3], gws[3];
  // Map of variable names and their values use to evaluate
  // the kernel launch parameters.
  CMapBasicBasic *map;
  // Boolean flag to determine if the grid size should be evaluated or not.
  int eval_grid;
  // Pointer to keep track of backend specific data.
  void *bptr;
  // Reduction related metadata.
  int redn_idx, redn_op, redn_type, redn_size;
  void *redn_ptr;
};

/**
 * @ingroup nomp_internal_types
 *
 * @brief Structure to keep track of nomp runtime data and backend specific
 * data. This structure is also used to dispatch backend specific functions.
 */
struct nomp_backend_t {
  /**
   * ID of the platform to be used for the backend. This is only used when the
   * backend is OpenCL.
   */
  int platform_id;
  /**
   * ID of the dhvice to be used with the backend.
   */
  int device_id;
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
   * Directory where transform and annotations cripts are located.
   */
  char scripts_dir[PATH_MAX + 1];

  /**
   * Function pointer to the backend update function which can allocate,
   * update and free backend memory.
   */
  int (*update)(struct nomp_backend_t *, struct nomp_mem_t *,
                const nomp_map_direction_t op, size_t start, size_t end,
                size_t usize);
  /**
   * Function pointer to the backend kernel build function.
   */
  int (*knl_build)(struct nomp_backend_t *, struct nomp_prog_t *, const char *,
                   const char *);
  /**
   * Function pointer to the backend kernel run function.
   */
  int (*knl_run)(struct nomp_backend_t *, struct nomp_prog_t *);
  /**
   * Function pointer to the backend kernel free function.
   */
  int (*knl_free)(struct nomp_prog_t *);
  /**
   * Function pointer to the backend synchronization function.
   */
  int (*sync)(struct nomp_backend_t *);
  /**
   * Function pointer to the backend finalize function which releases allocated
   * resources.
   */
  int (*finalize)(struct nomp_backend_t *);

  /**
   * Scratch memory to be used as temporary memory for kernels (like reductions)
   */
  struct nomp_mem_t scratch;

  /**
   * Python function object which will be called to perform annotations.
   */
  PyObject *py_annotate;

  /**
   * Context info is used to pass necessary infomation to kernel transformations
   * and annotations.
   */
  PyObject *py_context;

  /**
   * Pointer to keep track of backend specific data. This is allocated and
   * released by the backend.
   */
  void *bptr;
};

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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_backend_init Backend init functions
 *
 * @brief Functions for initializing different backend as OpenCL, CUDA, etc.
 * These has to be refactored to handle the case when the backend is not
 * available.
 */

int opencl_init(struct nomp_backend_t *backend, const int platform_id,
                const int device_id);

int cuda_init(struct nomp_backend_t *backend, const int platform_id,
              const int device_id);

int hip_init(struct nomp_backend_t *backend, const int platform_id,
             const int device_id);

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_IMPL_H_
