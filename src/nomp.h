#if !defined(_LIB_NOMP_H_)
#define _LIB_NOMP_H_

#include <stddef.h>
/**
 * @defgroup nomp_update_direction Update Direction
 *
 * @brief Defines the update direction (or operation) in nomp_update().
 */

/**
 * @ingroup nomp_update_direction
 * @brief Allocate memory on the device.
 */
#define NOMP_ALLOC 1
/**
 * @ingroup nomp_update_direction
 * @brief Copy host data to device. Memory will be allocated if not allocated
 * already.
 */
#define NOMP_TO 2
/**
 * @ingroup nomp_update_direction
 * @brief Copy device data to host.
 */
#define NOMP_FROM 4
/**
 * @ingroup nomp_update_direction
 * @brief Free memory allocated on the device.
 */
#define NOMP_FREE 8

/**
 * @defgroup nomp_types Data Types
 *
 * @brief Defines argument type in a kernel. Currently, only integer, float or
 * pointer types are supported.
 */

/**
 * @ingroup nomp_types
 * @brief Integer argument type.
 */
#define NOMP_INTEGER 1
/**
 * @ingroup nomp_types Data Types
 * @brief Floating point argument type.
 */
#define NOMP_FLOAT 2
/**
 * @ingroup nomp_types
 * @brief Pointer argument type.
 */
#define NOMP_PTR 4

/**
 * @defgroup nomp_errors Errors
 *
 * @brief Different types of errors returned by libnomp api calls.
 */

/**
 * @ingroup nomp_errors
 * @brief Invalid libnomp backend
 */
#define NOMP_INVALID_BACKEND -32
/**
 * @ingroup nomp_errors
 * @brief Invalid libnomp platform
 */
#define NOMP_INVALID_PLATFORM -33
/**
 * @ingroup nomp_errors
 * @brief Invalid libnomp device
 */
#define NOMP_INVALID_DEVICE -34
/**
 * @ingroup nomp_errors
 * @brief Invalid libnomp map pointer
 */
#define NOMP_INVALID_MAP_PTR -36
/**
 * @ingroup nomp_errors
 * @brief Invalid libnomp map operation
 */
#define NOMP_INVALID_MAP_OP -37
/**
 * @ingroup nomp_errors
 * @brief Pointer is already mapped
 */
#define NOMP_PTR_ALREADY_MAPPED -38
/**
 * @ingroup nomp_errors
 * @brief Invalid libnomp kernel
 */
#define NOMP_INVALID_KNL -39
/**
 * @ingroup nomp_errors
 * @brief Invalid libnomp for clause
 */
#define NOMP_INVALID_CLAUSE -40
/**
 * @ingroup nomp_errors
 * @brief Invalid libnomp map parameters
 */
#define NOMP_INVALID_MAP_PARAMS -41

/**
 * @ingroup nomp_errors
 * @brief libnomp is already initialized
 */
#define NOMP_INITIALIZED_ERROR -64
/**
 * @ingroup nomp_errors
 * @brief libnomp is not initialized
 */
#define NOMP_NOT_INITIALIZED_ERROR -65
/**
 * @ingroup nomp_errors
 * @brief Failed to finalize libnomp
 */
#define NOMP_FINALIZE_ERROR -66
/**
 * @ingroup nomp_errors
 * @brief libnomp memory free failed
 */
#define NOMP_FREE_FAILURE -67
/**
 * @ingroup nomp_errors
 * @brief libnomp malloc failed
 */
#define NOMP_MALLOC_FAILURE -68
/**
 * @ingroup nomp_errors
 * @brief libnomp realloc failed
 */
#define NOMP_REALLOC_FAILURE -69

/**
 * @ingroup nomp_errors
 * @brief libnomp python initialization failed
 */
#define NOMP_PY_INITIALIZE_ERROR -96
/**
 * @ingroup nomp_errors
 * @brief NOMP_INSTALL_DIR env. variable is not set
 */
#define NOMP_INSTALL_DIR_NOT_FOUND -97
/**
 * @ingroup nomp_errors
 * @brief Specified user callback function not found
 */
#define NOMP_USER_CALLBACK_NOT_FOUND -98
/**
 * @ingroup nomp_errors
 * @brief User callback function failed
 */
#define NOMP_USER_CALLBACK_FAILURE -99
/**
 * @ingroup nomp_errors
 * @brief Loopy conversion failed
 */
#define NOMP_LOOPY_CONVERSION_ERROR -100
/**
 * @ingroup nomp_errors
 * @brief Failed to find loopy kernel
 */
#define NOMP_LOOPY_KNL_NAME_NOT_FOUND -101
/**
 * @ingroup nomp_errors
 * @brief Code generation from loopy kernel failed
 */
#define NOMP_LOOPY_CODEGEN_FAILED -102
/**
 * @ingroup nomp_errors
 * @brief Code generation from loopy kernel failed
 */
#define NOMP_GET_GRIDSIZE_FAILED -103
/**
 * @ingroup nomp_errors
 * @brief Grid size calculation failed
 */
#define NOMP_EVAL_GRIDSIZE_FAILED -103
/**
 * @ingroup nomp_errors
 * @brief libnomp python initialization failed
 */
#define NOMP_PY_APPEND_PATH_ERROR -104

/**
 * @ingroup nomp_errors
 * @brief libnomp kernel build failed
 */
#define NOMP_KNL_BUILD_ERROR -128
/**
 * @ingroup nomp_errors
 * @brief Invalid libnomp kernel argument type
 */
#define NOMP_KNL_ARG_TYPE_ERROR -129
/**
 * @ingroup nomp_errors
 * @brief Setting libnomp kernel argument failed
 */
#define NOMP_KNL_ARG_SET_ERROR -130
/**
 * @ingroup nomp_errors
 * @brief libnomp kernel run failed
 */
#define NOMP_KNL_RUN_ERROR -131
/**
 * @ingroup nomp_errors
 * @brief libnomp run out of memory
 */
#define NOMP_OUT_OF_MEMORY -140
/**
 * @ingroup nomp_errors
 * @brief libnomp invalid log id
 */
#define NOMP_INVALID_LOG_ID -141
/**
 * @ingroup nomp_errors
 * @brief libnomp unknown error
 */
#define NOMP_UNKNOWN_ERROR -142
/**
 * @ingroup nomp_errors
 * @brief libnomp string length exceed max length.
 */
#define NOMP_STR_EXCEED_MAX_LEN -143
/**
 * @ingroup nomp_errors
 * @brief libnomp Cuda operation failed.
 */
#define NOMP_CUDA_FAILURE -144
/**
 * @ingroup nomp_errors
 * @brief libnomp tcalloc failure.
 */
#define NOMP_TCALLOC_FAILED -145
/**
 * @ingroup nomp_errors
 * @brief libnomp OpenCL failure.
 */
#define NOMP_OPENCL_FAILURE -146

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_user_api User API functions
 *
 * @brief libnomp API functions defined in `nomp.h`.
 */

/**
 * @ingroup nomp_user_api
 * @brief Initializes libnomp with the specified backend, platform and device.
 *
 * @details Initializes nomp code generation for the specified backend (e.g.,
 * OpenCL, Cuda, etc) using the given platform id and device id. Returns a
 * negative value if an error occurs during the initialization, otherwise
 * returns 0. Calling this method twice (without nomp_finalize in between) will
 * return an error (but not segfault). Currently only supports Cuda and OpenCL
 * backends.
 *
 * @param[in] backend Target backend for code generation.
 * @param[in] platform Target platform id (only used for OpenCL backend).
 * @param[in] device Target device id to use for computation.
 * @return int
 *
 * <b>Example usage:</b>
 * @code{.c}
 * int err = nomp_init("OpenCL", 0, 0);
 * @endcode
 */
int nomp_init(const char *backend, int platform, int device);

/**
 * @ingroup nomp_user_api
 * @brief Performs device to host (D2H) and host to device (H2D) memory
 * transfers, allocating and freeing of memory in the device.
 *
 * @param[in] ptr Pointer to the host memory location.
 * @param[in] start_idx Start index in the vector to start copying.
 * @param[in] end_idx End index in the vector to end the copying.
 * @param[in] unit_size Size of a single vector element.
 * @param[in] op Operation to perform (One of @ref nomp_update_direction).
 * @return int
 *
 * @details Operation \p op will be performed on the array slice [\p start_idx,
 * \p end_idx), i.e., on array elements start_idx, ... end_idx - 1. This method
 * returns a non-zero value if there is an error and 0 otherwise.
 *
 * <b>Example usage:</b>
 * @code{.c}
 * int N = 10;
 * double a[10];
 * for (unsigned i = 0; i < N; i++)
 *   a[i] = i;
 *
 * // Copy the value of `a` into device
 * int err = nomp_update(a, 0, N, sizeof(double), NOMP_TO);
 *
 * // Execution of a kernel which uses `a`
 * ...
 *
 * // Copy the updated value of `a` from device
 * int err = nomp_update(a, 0, N, sizeof(double), NOMP_FROM);
 *
 * // Free the device memory allocated for `a`
 * int err = nomp_update(a, 0, N, sizeof(double), NOMP_FREE);
 *
 * @endcode
 */
int nomp_update(void *ptr, size_t start_idx, size_t end_idx, size_t unit_size,
                int op);

/**
 * @ingroup nomp_user_api
 * @brief Generate and compile a kernel for the targe backend (OpenCL, etc.)
 * from C source.
 *
 * @details Target backend is the one provided during the initialization of
 * libnomp using nomp_init(). User defined code transformations will be applied
 * based on the clauses specified in \p clauses argument. Additional kernel meta
 * data can be passed using the \p annotations argument.
 *
 * <b>Example usage:</b>
 * @code{.c}
 * int N = 10;
 * double a[10], b[10];
 * for (unsigned i = 0; i < N; i++) {
 *   a[i] = i;
 *   b[i] = 10 -i
 * }
 * const char *knl = "for (unsigned i = 0; i < N; i++) a[i] += b[i];"
 * static int id = -1;
 * const char *annotations[1] = {0},
 *            *clauses[3] = {"transform", "file:function", 0};
 * int err = nomp_jit(&id, knl, annotations, clauses);
 * @endcode
 *
 * @param[out] id Id of the generated kernel.
 * @param[in] c_src Kernel source in C.
 * @param[in] annotations Annotations to perform user defined transformations.
 * @param[in] clauses Clauses to provide meta information about the kernel.
 * @return int
 */
int nomp_jit(int *id, const char *c_src, const char **annotations,
             const char **clauses);

/**
 * @ingroup nomp_user_api
 * @brief Runs the kernel generated by nomp_jit().
 *
 * @details Runs the kernel with a given kernel id. Kernel id is followed by the
 * number of arguments (i.e., variables). Then for each argument, four values
 * has to be passed. First, is the argument name as a string. Second is the
 * argument type (one of @ref nomp_types). Third is the `sizeof` argument and
 * the fourth is the pointer to the actual argument itself.
 *
 * @param[in] id Id of the kernel to be run.
 * @param[in] nargs Number of arguments to the kernel.
 * @param[in] ...  Four values mentioned above for each argument.
 *
 * @return int
 */
int nomp_run(int id, int nargs, ...);

void nomp_assert_(int cond, const char *file, unsigned line);
#define nomp_assert(cond) nomp_assert_(cond, __FILE__, __LINE__)

void nomp_chk_(int err, const char *file, unsigned line);
#define nomp_chk(err) nomp_chk_(err, __FILE__, __LINE__)

/**
 * @ingroup nomp_logs
 * @brief NOMP logs can be of an error, warning or an information.
 */
typedef enum {
  NOMP_ERROR = 0,
  NOMP_WARNING = 1,
  NOMP_INFORMATION = 2,
  NOMP_INVALID = 3
} nomp_log_type;

/**
 * @ingroup nomp_user_api
 * @brief Return log description.
 *
 * @details Returns the log description given the log_id.
 * @param[out] log variable to set the error description.
 * @param[in] log_id id of the error.
 * @return int
 */
int nomp_get_log_str(char **log, int log_id);

/**
 * @ingroup nomp_user_api
 * @brief Return log number.
 *
 * @details Returns the log number given the log_id. If log_id
 * is invalid return NOMP_INVALID_LOG_ID.
 * @param[in] log_id id of the log.
 * @return int
 */
int nomp_get_log_no(int log_id);

/**
 * @ingroup nomp_user_api
 * @brief Return log type.
 *
 * @details Returns the log type given the log_id. Log type is either
 * NOMP_ERROR, NOMP_INFORMATION or NOMP_WARNING. If log_id is invalid return
 * NOMP_INVALID.
 * @param[in] log_id id of the log.
 * @return int
 */
nomp_log_type nomp_get_log_type(int log_id);

/**
 * @ingroup nomp_user_api
 * @brief Finalizes libnomp runtime.
 *
 * @details Frees allocated runtime resources for libnomp. Returns a non-zero
 * value if an error occurs during the finalize process, otherwise returns 0.
 * Calling this method before `nomp_init` will retrun an error. Calling this
 * method twice will also return an error.
 *
 * @return int
 */
int nomp_finalize(void);
#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_H_
