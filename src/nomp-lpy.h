#if !defined(_NOMP_LPY_H_)
#define _NOMP_LPY_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Forward declare the `struct nomp_prog`.
struct nomp_prog;

#ifdef __cplusplus
extern "C" {
#endif

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
int nomp_py_append_to_sys_path(const char *path);

/**
 * @ingroup nomp_py_utils
 * @brief Creates loopy kernel from C source.
 *
 * @param[out] knl Loopy kernel object.
 * @param[in] src C kernel source.
 * @param[in] backend Backend name.
 * @return int
 */
int nomp_py_c_to_loopy(PyObject **knl, const char *src, const char *backend);

/**
 * @ingroup nomp_py_utils
 * @brief Realize reductions if present in the kernel.
 *
 * @param[in,out] knl Loopy kernel object.
 * @param[in] var Name of the reduction variable.
 * @return int
 */
int nomp_py_realize_reduction(PyObject **knl, const char *var);

/**
 * @ingroup nomp_py_utils
 * @brief Set the annotate function based on the path to annotation script and
 * function.
 *
 * @param[out] func Pointer to the annotate function.
 * @param[in] path Path to the annotation script followed by function name (path
 * and function name must be separated by "::").
 * @return int
 */
int nomp_py_set_annotate_func(PyObject **func, const char *path);

/**
 * @ingroup nomp_py_utils
 * @brief Apply transformations on a loopy kernel based on annotations.
 *
 * Apply the transformations to the loopy kernel \p knl based on the annotation
 * function \p func and the key value pairs (annotations) passed in \p annts.
 * \p knl will be modified based on the transformations.
 *
 * @param[in,out] knl Pointer to loopy kernel object.
 * @param[in] func Function which performs transformations based on annotations.
 * @param[in] annts Annotations (as a PyDict) to specify which transformations
 * to apply.
 * @param[in] context Context (as a PyDict) to pass around information such
 * as backend, device details, etc.
 * @return int
 */
int nomp_py_apply_annotations(PyObject **knl, PyObject *func,
                              const PyObject *annts, const PyObject *context);

/**
 * @ingroup nomp_py_utils
 * @brief Apply kernel specific user transformations on a loopy kernel.
 *
 * Call the user transform function \p func in file \p file on the loopy
 * kernel \p knl. \p knl will be modified based on the transformations.
 * Function will return a non-zero value if there was an error after
 * registering a log.
 *
 * @param[in,out] knl Pointer to loopy kernel object.
 * @param[in] file Path to the file containing transform function \p func.
 * @param[in] func Transform function.
 * @param[in] context Context (as a PyDict) to pass around information such
 * as backend, device details, etc.
 * @return int
 */
int nomp_py_apply_transform(PyObject **knl, const char *file, const char *func,
                            const PyObject *context);

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
int nomp_py_get_knl_name_and_src(char **name, char **src, const PyObject *knl,
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
int nomp_py_get_grid_size(struct nomp_prog *prg, PyObject *knl);

/**
 * @ingroup nomp_py_utils
 * @brief Evaluate global and local grid sizes based on the dictionary `dict`.
 *
 * @param[in] prg Nomp program.
 * @return int
 */
int nomp_py_eval_grid_size(struct nomp_prog *prg);

/**
 * @ingroup nomp_py_utils
 * @brief Map the keys and values to evaluate the kernel launch parameters.
 *
 * @param[in] prg Nomp program.
 * @param[in] key Key as a C-string.
 * @param[in] val Value as a C-string.
 * @return int
 */
int sym_c_map_push(struct prog *prg, const char *key, const char *val);

/**
 * @ingroup nomp_py_utils
 * @brief Get the string representation of python object.
 *
 * @param msg Debug message before printing the object.
 * @param obj Python object.
 * @return void
 */
void nomp_py_print(const char *msg, PyObject *obj);

#ifdef __cplusplus
}
#endif

#endif // _NOMP_LPY_H_
