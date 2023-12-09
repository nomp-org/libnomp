#if !defined(_NOMP_LOOPY_H_)
#define _NOMP_LOOPY_H_

#include "nomp-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_py_utils Python helper functions
 *
 * @brief Python helper functions for calling loopy and other python functions.
 */

int nomp_py_init(const nomp_config_t *cfg);

int nomp_py_append_to_sys_path(const char *path);

int nomp_py_check_module(const char *module, const char *function);

int nomp_py_c_to_loopy(PyObject **knl, const char *src);

int nomp_py_realize_reduction(PyObject **knl, const char *var,
                              const PyObject *context);

int nomp_py_transform(PyObject **knl, const char *file, const char *func,
                      const PyObject *context);

int nomp_py_get_knl_name_and_src(char **name, char **src, const PyObject *knl);

int nomp_py_set_annotate_func(PyObject **func, const char *path);

int nomp_py_annotate(PyObject **kernel, PyObject *function,
                     const PyObject *annotations, const PyObject *context);

int nomp_py_get_grid_size(nomp_prog_t *prg, PyObject *knl);

int nomp_py_fix_parameters(PyObject **knl, const PyObject *py_dict);

void nomp_py_get_str(const char *msg, PyObject *obj);

int nomp_py_finalize(void);

int nomp_symengine_eval_grid_size(nomp_prog_t *prg);

int nomp_symengine_update(CMapBasicBasic *map, const char *key, const long val);

#ifdef __cplusplus
}
#endif

#endif // _NOMP_LOOPY_H_
