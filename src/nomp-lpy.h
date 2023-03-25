#if !defined(_NOMP_LPY_H_)
#define _NOMP_LPY_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Forward declare the `struct prog`.
struct prog;

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
 * @param[in] src C kernel source.
 * @param[in] backend Backend name.
 * @return int
 */
int py_c_to_loopy(PyObject **knl, const char *src, const char *backend);

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
 * @param[in] backend Backend name.
 * @return int
 */
int py_get_knl_name_and_src(char **name, char **src, const PyObject *knl,
                            const char *backend);

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

#endif // _NOMP_LPY_H_
