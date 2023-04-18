#if !defined(_NOMP_REDUCTION_H_)
#define _NOMP_REDUCTION_H_

#include "nomp-impl.h"

/**
 * @defgroup nomp_reduction_ops Reduction operations supported by nomp.
 *
 * @brief Defines reduction operations allowed in nomp kernels.
 */

/**
 * @ingroup nomp_reduction_op
 * @brief Sum reduction operation.
 */
#define NOMP_SUM 0
/**
 * @ingroup nomp_reduction_op
 * @brief Product reduction operation.
 */
#define NOMP_PROD 1

/**
 * @ingroup nomp_reduction_utils
 * @brief Perform host side reduction.
 *
 * @param[in] bnd Active backend instance.
 * @param[in] prg Active program instance.
 * @param[in] m Memory used to store device side partial reductions.
 * @return int
 */
int host_side_reduction(struct backend *bnd, struct prog *prg, struct mem *m);

/**
 * @ingroup nomp_py_utils
 * @brief Get kernal name and generated source for the backend.
 * @brief Handle reductions (if present) in a loopy kernel.
 *
 * Handle the reduction in the loopy kernel \p knl. \p knl will be modified
 * in the process. Function will return a non-zero value if there was an error
 * after registering a log.
 *
 * @param[in,out] knl Pointer to loopy kernel object.
 * @param[out] redn_op Reduction operation.
 * @param[in] backend Backend for the reduction.
 * @return int
 */
int py_handle_reduction(PyObject **knl, int *redn_op, const char *backend);

#endif
