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

#define py_dir "python"
#define py_module "loopy_api"
#define py_func "c_to_loopy"

#define FREE(x)                                                                \
  do {                                                                         \
    if (x)                                                                     \
      free(x);                                                                 \
  } while (0)

#define return_on_err(err, ...)                                                \
  do {                                                                         \
    if (err)                                                                   \
      return err;                                                              \
  } while (0)

#define tmalloc(type, count) ((type *)malloc((count) * sizeof(type)))
#define tcalloc(type, count) ((type *)calloc((count), sizeof(type)))
#define trealloc(type, ptr, count)                                             \
  ((type *)realloc((ptr), (count) * sizeof(type)))

struct prog {
  unsigned nargs, ndim;
  size_t local[3], global[3];
  void *bptr;
};

struct mem {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
};

struct backend {
  char name[BUFSIZ];
  int (*map)(struct backend *, struct mem *, const int);
  int (*knl_build)(struct backend *, struct prog *, const char *, const char *);
  int (*knl_run)(struct backend *, struct prog *, va_list);
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
 * @param platform_id Target platform id to share resources and execute kernals
 *                 in the targeted device.
 * @param device_id Target device id to execute kernals.
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
 * @param platform_id Target platform id to share resources and execute kernals
 *                 in the targeted device.
 * @param device_id Target device id to execute kernals.
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
int py_get_grid_size(unsigned *ndim, size_t *global, size_t *local,
                     PyObject *pKnl, PyObject *pDict);

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

/**
 * @ingroup nomp_other_utils
 * @brief Convert a C-string to lowercase
 *
 * Convert input string `in` to lower case and store in `out`. Maximum size for
 * input string `in` is specified by `max`. Returns 0 if successful, otherwise
 * return 1.
 *
 * @param[out] out Address of output string
 * @param[in] in Input string
 * @param[in] max Maximum allowed size for the input string
 * @return int
 */
int strnlower(char **out, const char *in, size_t max);

#endif // _LIB_NOMP_IMPL_H_
