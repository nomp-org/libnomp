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

#include "nomp-log.h"
#include "nomp-mem.h"
#include "nomp.h"

#define NOMP_BUFSIZ 64
#define MAX_BACKEND_NAME_SIZE 32

#define return_on_err(err)                                                     \
  {                                                                            \
    if (nomp_get_log_type((err)) == NOMP_ERROR)                                \
      return (err);                                                            \
  }

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
  char *backend, *install_dir, *annts_script, *annts_func;
  int platform_id, device_id, verbose;
  char name[NOMP_BUFSIZ];
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
 * @param[out] knl Loopy kernel object.
 * @param[in] c_src C kernel source.
 * @param[in] backend Backend name.
 * @return int
 */
int py_c_to_loopy(PyObject **knl, const char *c_src, const char *backend);

/**
 * @ingroup nomp_py_utils
 * @brief Apply transformations on a loopy kernel based on annotations.
 *
 * Apply the transformations defined in function \p func in file \p file on the
 * loopy kernel \p knl based on the key value pairs (annotations) passed in \p
 * annts. \p knl will be modified based on the transformations. Function will
 * return a non-zero value if there was an error after registering a log.
 *
 * @param[in,out] knl Pointer to loopy kernel object.
 * @param[in] annts Annotations (as a PyDict) to specify which transformations
 * to apply.
 * @param[in] file Path to the file containing transform function \p func.
 * @param[in] func Transform function.
 * @return int
 */
int py_user_annotate(PyObject **knl, PyObject *annts, const char *file,
                     const char *func);

/**
 * @ingroup nomp_py_utils
 * @brief Apply kernel specific user transformations on a loopy kernel.
 *
 * Call the user transform function \p func in file \p file on the loopy kernel
 * \p knl. \p knl will be modified based on the transformations. Function will
 * return a non-zero value if there was an error after registering a log.
 *
 * @param[in,out] knl Pointer to loopy kernel object.
 * @param[in] file Path to the file containing transform function \p func.
 * @param[in] func Transform function.
 * @return int
 */
int py_user_transform(PyObject **knl, const char *file, const char *func);

/**
 * @ingroup nomp_py_utils
 * @brief Get kernel name and generated source for the backend.
 *
 * @param[out] name Kernel name as a C-string.
 * @param[out] src Kernel source as a C-string.
 * @param[in] knl Loopy kernel object.
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
 * @param[in] knl Python kernel object.
 * @return int
 */
int py_get_grid_size(struct prog *prg, PyObject *knl);

/**
 * @ingroup nomp_py_utils
 * @brief Evaluate global and local grid sizes based on the dictionary `dict`.
 *
 * @param[in] prg Nomp program.
 * @param[in] dict Dictionary with variable name as keys, variable value as
 * values.
 * @return int
 */
int py_eval_grid_size(struct prog *prg, PyObject *dict);

/**
 * @ingroup nomp_py_utils
 * @brief Get the string representation of python object.
 *
 * @param msg Debug message before printing the object.
 * @param obj Python object.
 * @return void
 */
void py_print(const char *msg, PyObject *obj);

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
 * @param[in] nstr Number of strings to concatenate.
 * @param[in] max_len Maximum length of an individual string.
 * @param[in] ... Strings to concatenate.
 * @return char*
 */
char *strcatn(unsigned nstr, unsigned max_len, ...);

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

/**
 * @ingroup nomp_other_utils
 * @brief Convert a string to unsigned long value if possible.
 *
 * Convert input string `str` to an unsigned int value. Returns converted
 * unsigned int value if successful, otherwise return -1.
 *
 * @param[in] str String to convert into unsigned int.
 * @param[in] size Length of the string.
 * @return int
 */
int strntoui(const char *str, size_t size);

/**
 * @ingroup nomp_other_utils
 * @brief Returns maximum length of a path.
 *
 * Returns the maximum length of specified path.
 *
 * @param[in] path Path to get the maximum length.
 * @return size_t
 */
size_t pathlen(const char *path);

/**
 * @ingroup nomp_other_utils
 * @brief Returns maximum among all integers passed.
 *
 * Returns the maximum between two or more integers.
 *
 * @param[in] args Total number of integers.
 * @param[in] ... List of integers to find the maximum of as a variable argument
 * list.
 * @return int
 */
int MAX(unsigned args, ...);

/**
 * @ingroup nomp_internal_api
 * @brief Returns a non-zero error if the input is NULL.
 *
 * This function call set_log() to register an error if the input is NULL.
 * Use the macro nomp_null_input() to automatically add last three arguments.
 *
 * @param[in] p Input pointer.
 * @param[in] func Function in which the null check is done.
 * @param[in] line Line number where the null check is done.
 * @param[in] file File name in which the null check is done.
 * @return int
 */
int check_null_input_(void *p, const char *func, unsigned line,
                      const char *file);
#define check_null_input(p)                                                    \
  return_on_err(check_null_input_((void *)(p), __func__, __LINE__, __FILE__))

/**
 * @ingroup nomp_log_utils
 * @brief Free log variables.
 *
 * @return void
 */
void finalize_logs();

#endif // _LIB_NOMP_IMPL_H_
