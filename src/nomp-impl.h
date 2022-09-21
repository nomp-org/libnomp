#if !defined(_LIB_NOMP_IMPL_H_)
#define _LIB_NOMP_IMPL_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nomp.h"

#define MAX_BACKEND_NAME_SIZE 32
#define py_dir "python"
#define py_module "loopy_api"
#define py_func "c_to_loopy"

struct prog {
  void *bptr;
};

struct mem {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
};

struct error {
  char *description;
  int type;
};

struct backend {
  char name[MAX_BACKEND_NAME_SIZE];
  int (*map)(struct backend *, struct mem *, const int);
  int (*knl_build)(struct backend *, struct prog *, const char *, const char *);
  int (*knl_run)(struct backend *, struct prog *, const int, const size_t *,
                 const size_t *, int, va_list);
  int (*knl_free)(struct prog *);
  int (*finalize)(struct backend *);
  void *bptr;
};

/**
 * @ingroup nomp_other_utils
 * @brief Returns the pointer to the allocated memory corresponding to 'p'.
 *
 * Returns the pointer to the allocated memory corresponding to 'p'.
 * If no buffer has been allocated for 'p' returns *mems_n*
 *
 * @param p Value which required the pointer of it
 * @return struct mem*
 */
struct mem *mem_if_mapped(void *p);

//==============================================================================
// Backend init functions
//

/**
 * @defgroup nomp_backend_init Backend init functions
 */

/**
 * @ingroup nomp_backend_init
 * @brief Initializes OpenCL backend with the specified platform and device
 *
 * Initializes OpenCL backend while creating the command queue using the
 * given platform id and device id. Returns a negative value if an error
 * occurs during the initialization, otherwise returns 0.
 *
 * @param backend Target backend for code generation.
 * @param platform Target platform id to share resources and execute kernals
 *                 in the targeted device.
 * @param device Target device id to execute kernals.
 * @return int
 */
int opencl_init(struct backend *backend, const int platform_id,
                const int device_id);
/**
 * @ingroup nomp_backend_init
 * @brief Initializes Cuda backend with the specified platform and device
 *
 * Initializes Cuda backend using the given platform id and device id.
 * Returns a negative value if an error occurs during the initialization,
 * otherwise returns 0.
 *
 * @param backend Target backend for code generation.
 * @param platform Target platform id to share resources and execute kernals
 *                 in the targeted device.
 * @param device Target device id to execute kernals.
 * @return int
 */
int cuda_init(struct backend *backend, const int platform_id,
              const int device_id);

//==============================================================================
// Python helper functions
//

/**
 * @defgroup nomp_py_utils Python helper functions
 */

/**
 * @ingroup nomp_py_utils
 * @brief Appends specified path to system path.
 *
 * @param path Path to append
 * @return int
 */
int py_append_to_sys_path(const char *path);
/**
 * @ingroup nomp_py_utils
 * @brief Creates loopy kernel from C source
 *
 * @param pKnl Python kernal object
 * @param c_src C kernal source
 * @param backend Backend name
 * @return int
 */
int py_c_to_loopy(PyObject **pKnl, const char *c_src, const char *backend);
/**
 * @ingroup nomp_py_utils
 * @brief Calls the user callback function specfied by file name and
 * function name, on the kernal
 *
 * @param pKnl Python kernal object
 * @param file File to the callback function
 * @param func Declared name of the callback function
 * @return int
 */
int py_user_callback(PyObject **pKnl, const char *file, const char *func);
/**
 * @ingroup nomp_py_utils
 * @brief Get kernal name and source
 *
 * @param name Array of pointers to name
 * @param src Array of pointers to kernal source
 * @param pKnl Python kernal object
 * @return int
 */
int py_get_knl_name_and_src(char **name, char **src, PyObject *pKnl);
/**
 * @ingroup nomp_py_utils
 * @brief Get global and local grid sizes
 *
 * @param ndim Number of dimensions of the kernel
 * @param global Global grid
 * @param local Local grid
 * @param pKnl Python kernal object
 * @param pDict Dictionary with variable name as keys, variable values as values
 * @return int
 */
int py_get_grid_size(int *ndim, size_t *global, size_t *local, PyObject *pKnl,
                     PyObject *pDict);

//==============================================================================
// Other helper functions
//
/**
 * @defgroup nomp_other_utils Other helper functions
 */

/**
 * @ingroup nomp_other_utils
 * @brief Concatenates atmost `nstr` strings.
 *
 * Concatenates atmost `nstr` strings and returns a pointer to
 * the destination.
 *
 * @param nstr Number of strings to concatenate
 * @param ... Strings to concatenate
 * @return char*
 */
char *strcatn(int nstr, ...);

#endif // _LIB_NOMP_IMPL_H_
