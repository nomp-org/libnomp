#if !defined(_LIB_NOMP_H_)
#define _LIB_NOMP_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_user_types User types
 *
 * @brief Below is a list of data types that are used in the user API.
 */

/**
 * @ingroup nomp_user_types
 *
 * @brief Defines argument types supported for a nomp kernel. Used in nomp_jit()
 * which is used for kernel creation. Currently, only integer, float or pointer
 * types are supported as kernel arguments.
 */
typedef enum {
  NOMP_INT = 2048,   /*!< Signed integer argument type.*/
  NOMP_UINT = 4096,  /*!< Unsigned integer argument type.*/
  NOMP_FLOAT = 8192, /*!< Floating point argument type.*/
  NOMP_PTR = 16384   /*!< Pointer argument type.*/
} nomp_arg_type_t;

/**
 * @ingroup nomp_user_types
 * @brief Defines the update method (operation) in nomp_update().
 */
typedef enum {
  NOMP_ALLOC = 1, /*!< Allocate memory on the device.*/
  NOMP_TO = 2,    /*!< Copy host data to device. Memory will be allocated if not
                   * allocated.*/
  NOMP_FROM = 4,  /*!< Copy device data to host.*/
  NOMP_FREE = 8   /*!< Free memory allocated on the device.*/
} nomp_map_direction_t;

/**
 * @ingroup nomp_user_types
 * @brief Defines various argument properties.
 */
typedef enum {
  NOMP_JIT = 1 /*!< Argument value is fixed when the kernel is generated. */
} nomp_arg_properties_t;

/**
 * @defgroup nomp_error_codes Error codes returned to the user
 *
 * @brief Error codes used by internal libnomp functions when calling
 * nomp_log() with ::NOMP_ERROR. User can query these error codes using
 * nomp_get_err_no() by passing the return value of a libnomp function call
 * in case of an error. \p NOMP_SUCCESS is used for error code when
 * nomp_log() is called with ::NOMP_WARNING or ::NOMP_INFO.
 */

/**
 * @ingroup nomp_error_codes
 *
 * @brief libnomp API call was successful.
 */
#define NOMP_SUCCESS 0
/**
 * @ingroup nomp_error_codes
 *
 * @brief One of the inputs to a libnomp function call are not valid.
 */
#define NOMP_USER_INPUT_IS_INVALID -128
/**
 * @ingroup nomp_error_codes
 *
 * @brief Map pointer provided to libnomp is not valid.
 */
#define NOMP_USER_MAP_PTR_IS_INVALID -130
/**
 * @ingroup nomp_error_codes
 *
 * @brief Map operation provided to libnomp is not applicable.
 */
#define NOMP_USER_MAP_OP_IS_INVALID -132
/**
 * @ingroup nomp_error_codes
 *
 * @brief Log id provided to libnomp is not valid.
 */
#define NOMP_USER_LOG_ID_IS_INVALID -134
/**
 * @ingroup nomp_error_codes
 *
 * @brief Kernel argument type provided to libnomp is not valid.
 */
#define NOMP_USER_KNL_ARG_TYPE_IS_INVALID -136

/**
 * @ingroup nomp_error_codes
 *
 * @brief libnomp is already initialized.
 */
#define NOMP_INITIALIZE_FAILURE -256
/**
 * @ingroup nomp_error_codes
 *
 * @brief Failed to finalize libnomp.
 */
#define NOMP_FINALIZE_FAILURE -258
/**
 * @ingroup nomp_error_codes
 *
 * @brief The feature is not implemented.
 */
#define NOMP_NOT_IMPLEMENTED_ERROR -260

/**
 *
 * @ingroup nomp_error_codes
 * @brief A python call made by libnomp failed.
 */
#define NOMP_PY_CALL_FAILURE -384
/**
 * @ingroup nomp_error_codes
 *
 * @brief Loopy conversion failed.
 */
#define NOMP_LOOPY_CONVERSION_FAILURE -386
/**
 * @ingroup nomp_error_codes
 *
 * @brief Failed to find loopy kernel.
 */
#define NOMP_LOOPY_KNL_NAME_NOT_FOUND -388
/**
 * @ingroup nomp_error_codes
 *
 * @brief Code generation from loopy kernel failed.
 */
#define NOMP_LOOPY_CODEGEN_FAILURE -390
/**
 * @ingroup nomp_error_codes
 *
 * @brief Code generation from loopy kernel failed.
 */
#define NOMP_LOOPY_GRIDSIZE_FAILURE -392
/**
 * @ingroup nomp_error_codes
 *
 * @brief libnomp CUDA operation failed.
 */
#define NOMP_CUDA_FAILURE -512
/**
 * @ingroup nomp_error_codes
 *
 * @brief libnomp HIP operation failed.
 */
#define NOMP_HIP_FAILURE -514
/**
 * @ingroup nomp_error_codes
 *
 * @brief libnomp OpenCL operation failed.
 */
#define NOMP_OPENCL_FAILURE -516

/**
 * @defgroup nomp_user_api User API
 *
 * @brief libnomp user API functions.
 */

int nomp_init(int argc, const char **argv);

int nomp_update(void *ptr, size_t start_index, size_t end_index,
                size_t unit_size, nomp_map_direction_t op);

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
