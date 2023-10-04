#if !defined(_NOMP_REDUCTION_H_)
#define _NOMP_REDUCTION_H_

#include "nomp-impl.h"

/**
 * @defgroup nomp_reduction_ops Reduction operations
 * @brief Defines reduction operations allowed in nomp kernels.
 */

/**
 * @ingroup nomp_reduction_ops
 * @def NOMP_SUM
 * @brief Sum reduction operation.
 */
#define NOMP_SUM 0
/**
 * @ingroup nomp_reduction_ops
 * @def NOMP_PROD
 * @brief Product reduction operation.
 */
#define NOMP_PROD 1

/**
 * @defgroup nomp_reduction_utils Reduction utilities
 * @brief Perform host side reductions.
 */
int nomp_host_side_reduction(nomp_backend_t *bnd, nomp_prog_t *prg,
                             nomp_mem_t *m);

#endif
