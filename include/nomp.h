#if !defined(_LIB_NOMP_H_)
#define _LIB_NOMP_H_

#include <stddef.h>

/**
 * @defgroup nomp_user_types User types
 * @brief Data types used in libnomp user API.
 */

/**
 * @ingroup nomp_user_types
 *
 * @brief Defines argument types for a nomp kernel. Currently, only integer,
 * float or pointer types are supported.
 */
typedef enum {
  NOMP_INT = 2048,   /*!< Signed integer argument type.*/
  NOMP_UINT = 4096,  /*!< Unsigned integer argument type.*/
  NOMP_FLOAT = 6144, /*!< Floating point argument type.*/
  NOMP_PTR = 8192    /*!< Pointer argument type.*/
} nomp_type_t;

/**
 * @ingroup nomp_user_types
 * @brief Defines the update method (or operation) in nomp_update().
 */
typedef enum {
  NOMP_ALLOC = 1, /*!< Allocate memory on the device.*/
  NOMP_TO = 2,    /*!< Copy host data to device. Memory will be allocated if not
                   * allocated.*/
  NOMP_FROM = 4,  /*!< Copy device data to host.*/
  NOMP_FREE = 8   /*!< Free memory allocated on the device.*/
} nomp_map_direction_t;

/**
 * @defgroup nomp_user_errors Error codes returned to the user.
 *
 * @brief Error codes returned by libnomp user API calls to the
 * user. These error codes are negative integers.
 */

/**
 * @ingroup nomp_user_errors
 * @brief One of the inputs to a libnomp function call are not valid.
 */
#define NOMP_USER_INPUT_IS_INVALID -128
/**
 * @ingroup nomp_user_errors
 * @brief Map pointer provided to libnomp is not valid.
 */
#define NOMP_USER_MAP_PTR_IS_INVALID -130
/**
 * @ingroup nomp_user_errors
 * @brief Map operation provided to libnomp is not applicable.
 */
#define NOMP_USER_MAP_OP_IS_INVALID -132
/**
 * @ingroup nomp_user_errors
 * @brief Log id provided to libnomp is not valid.
 */
#define NOMP_USER_LOG_ID_IS_INVALID -134
/**
 * @ingroup nomp_user_errors
 * @brief Kernel argument type provided to libnomp is not valid.
 */
#define NOMP_USER_KNL_ARG_TYPE_IS_INVALID -136

/**
 * @ingroup nomp_user_errors
 * @brief libnomp is already initialized.
 */
#define NOMP_INITIALIZE_FAILURE -256
/**
 * @ingroup nomp_user_errors
 * @brief Failed to finalize libnomp.
 */
#define NOMP_FINALIZE_FAILURE -258
/**
 * @ingroup nomp_user_errors
 * @brief The feature is not implemented.
 */
#define NOMP_NOT_IMPLEMENTED_ERROR -260

/**
 * @ingroup nomp_user_errors
 * @brief A python call made by libnomp failed.
 */
#define NOMP_PY_CALL_FAILURE -384
/**
 * @ingroup nomp_user_errors
 * @brief Loopy conversion failed.
 */
#define NOMP_LOOPY_CONVERSION_FAILURE -386
/**
 * @ingroup nomp_user_errors
 * @brief Failed to find loopy kernel.
 */
#define NOMP_LOOPY_KNL_NAME_NOT_FOUND -388
/**
 * @ingroup nomp_user_errors
 * @brief Code generation from loopy kernel failed.
 */
#define NOMP_LOOPY_CODEGEN_FAILURE -390
/**
 * @ingroup nomp_user_errors
 * @brief Code generation from loopy kernel failed.
 */
#define NOMP_LOOPY_GRIDSIZE_FAILURE -392
/**
 * @ingroup nomp_user_errors
 * @brief libnomp CUDA operation failed.
 */
#define NOMP_CUDA_FAILURE -512
/**
 * @ingroup nomp_user_errors
 * @brief libnomp HIP failed.
 */
#define NOMP_HIP_FAILURE -514
/**
 * @ingroup nomp_user_errors
 * @brief libnomp OpenCL failure.
 */
#define NOMP_OPENCL_FAILURE -516

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_user_api User API
 * @brief libnomp API functions defined in `nomp.h`.
 */

int nomp_init(int argc, const char **argv);

int nomp_update(void *ptr, size_t start_idx, size_t end_idx, size_t unit_size,
                nomp_map_direction_t op);

int nomp_jit(int *id, const char *src, const char **clauses, int nargs, ...);

int nomp_run(int id, ...);

int nomp_sync(void);

char *nomp_get_err_str(unsigned id);

int nomp_get_err_no(unsigned id);

int nomp_finalize(void);

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_H_
