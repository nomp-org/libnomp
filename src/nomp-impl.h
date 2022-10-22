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
  PyObject *py_global, *py_local, *py_dict;
  size_t global[3], local[3];
  void *bptr;
};

struct mem {
  size_t idx0, idx1, usize;
  void *hptr, *bptr;
};

struct backend {
  char name[BUFSIZ];
  int (*update)(struct backend *, struct mem *, const int);
  int (*knl_build)(struct backend *, struct prog *, const char *, const char *);
  int (*knl_run)(struct backend *, struct prog *, va_list);
  int (*knl_free)(struct prog *);
  int (*finalize)(struct backend *);
  void *bptr;
};

/**
 * @ingroup nomp_other_utils
 * @brief Returns the mem object corresponding to host pointer `p`.
 *
 * Returns the mem object corresponding to host ponter `p`. If no buffer has
 * been allocated for `p` on the device, returns NULL.
 *
 * @param[in] p Host pointer
 * @return struct mem *
 */
struct mem *mem_if_mapped(void *p);

/**
 * @defgroup nomp_backend_init Backend init functions
 */

/**
 * @ingroup nomp_backend_init
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
 * @defgroup nomp_py_utils Python helper functions
 */

/**
 * @ingroup nomp_py_utils
 * @brief Appends specified path to system path.
 *
 * @param[in] path Path to be appended to system path.
 * @return int
 */
int py_append_to_sys_path(const char *path);
/**
 * @ingroup nomp_py_utils
 * @brief Creates loopy kernel from C source.
 *
 * @param[out] knl Loopy kernal object.
 * @param[in] c_src C kernal source.
 * @param[in] backend Backend name.
 * @return int
 */
int py_c_to_loopy(PyObject **knl, const char *c_src, const char *backend);
/**
 * @ingroup nomp_py_utils
 * @brief Calls the user callback function `func` in file `file`.
 *
 * @param[in,out] knl Python kernal object.
 * @param[in] file File with the callback function.
 * @param[in] func Callback function name.
 * @return int
 */
int py_user_callback(PyObject **knl, const char *file, const char *func);
/**
 * @ingroup nomp_py_utils
 * @brief Get kernal name and generated source for the backend.
 *
 * @param[out] name Kernel name as a C-string.
 * @param[out] src Kernel source as a C-string.
 * @param[in] knl Loopy kernal object.
 * @return int
 */
int py_get_knl_name_and_src(char **name, char **src, PyObject *knl);
/**
 * @ingroup nomp_py_utils
 * @brief Get global and local grid sizes as `pymoblic` expressions.
 *
 * Grid sizes are stored in the program object itself.
 *
 * @param[in] prg Nomp program object.
 * @param[in] knl Python kernal object.
 * @return int
 */
int py_get_grid_size(struct prog *prg, PyObject *knl);
/**
 * @ingroup nomp_py_utils
 * @brief Evaluate global and local grid sizes based on the dictionaty `dict`.
 *
 * @param[in] prg Nomp program.
 * @param[in] dict Dictionary with variable name as keys, variable value as
 * values.
 * @return int
 */
int py_eval_grid_size(struct prog *prg, PyObject *dict);
/**
 * @ingroup nomp_py_utils
 * @brief Get the representation of python object.
 *
 * @param obj Python object.
 * @return void
 */
void py_print(PyObject *obj);

/**
 * @defgroup nomp_other_utils Other helper functions.
 */

/**
 * @ingroup nomp_other_utils
 * @brief Concatenates atmost `nstr` strings.
 *
 * Concatenates atmost `nstr` strings and returns a pointer to
 * resulting string.
 *
 * @param nstr Number of strings to concatenate.
 * @param ... Strings to concatenate.
 * @return char*
 */
char *strcatn(int nstr, ...);

/**
 * @ingroup nomp_other_utils
 * @brief Convert a C-string to lowercase
 *
 * Convert input string `in` to lower case and store in `out`. Maximum length
 * of the input string `in` is specified by `max`. Returns 0 if successful, 1
 * otherwise.
 *
 * @param[out] out Output string.
 * @param[in] in Input string.
 * @param[in] max Maximum allowed length for the input string.
 * @return int
 */
int strnlower(char **out, const char *in, size_t max);

#endif // _LIB_NOMP_IMPL_H_
