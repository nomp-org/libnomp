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
int nomp_host_side_reduction(struct nomp_backend *bnd, struct nomp_prog *prg,
                             struct nomp_mem *m);

#endif
