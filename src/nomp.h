#if !defined(_LIB_NOMP_H_)
#define _LIB_NOMP_H_

#include <stddef.h>

/**
 * @defgroup nomp_attributes Additional attributes for kernel arguments.
 *
 * @brief Defines additional attributes for nomp kernel arguments. For example,
 * these can be used to communicate if the argument is a reduction variable
 * and/or if the pinned memory must be used for the argument on the device.
 */

/**
 * @ingroup nomp_attributes
 * @brief Nomp kernel argument is an accumulator of a reduction.
 */
#define NOMP_ATTR_REDN 1
/**
 * @ingroup nomp_attributes
 * @brief Device memory for the Nomp kernel argument must be pinned.
 */
#define NOMP_ATTR_PINNED 2
/**
 * @ingroup nomp_attributes
 * @brief Device memory for the Nomp kernel argument should be pinned.
 */
#define NOMP_ATTR_MASK 2047

/**
 * @defgroup nomp_types Nomp data types.
 *
 * @brief Defines argument types for a nomp kernel. Currently, only integer,
 * float or pointer types are supported.
 */

/**
 * @ingroup nomp_types
 * @brief Signed integer argument type.
 */
#define NOMP_INT 2048
/**
 * @ingroup nomp_types
 * @brief Unsigned integer argument type.
 */
#define NOMP_UINT 4096
/**
 * @ingroup nomp_types
 * @brief Floating point argument type.
 */
#define NOMP_FLOAT 6144
/**
 * @ingroup nomp_types
 * @brief Pointer argument type.
 */
#define NOMP_PTR 8192

/**
 * @defgroup nomp_update_op Memory update operations for Nomp variable.
 *
 * @brief Defines update operations for a pointer type variable as passed into
 * nomp_update().
 */

/**
 * @ingroup nomp_update_op
 * @brief Allocate memory on the device.
 */
#define NOMP_ALLOC 1
/**
 * @ingroup nomp_update_op
 * @brief Copy host data to device. Memory will be allocated if necessary.
 */
#define NOMP_TO 2
/**
 * @ingroup nomp_update_op
 * @brief Copy device data to host. User must make sure that there is enough
 * memory available on the host.
 */
#define NOMP_FROM 4
/**
 * @ingroup nomp_update_op
 * @brief Free memory allocated on the device.
 */
#define NOMP_FREE 8

/**
 * @defgroup nomp_reduction_ops Nomp reduction operations.
 *
 * @brief Defines reduction operations supported by nomp.
 */

/**
 * @ingroup nomp_reduction_ops
 * @brief Sum reduction operation.
 */
#define NOMP_SUM 1
/**
 * @ingroup nomp_reduction_ops
 * @brief Production reduction operation.
 */
#define NOMP_PROD 2

/**
 * @defgroup nomp_errors Errors
 *
 * @brief Different types of errors returned by libnomp api calls.
 */

/**
 * @ingroup nomp_errors
 * @brief One of the inputs to a libnomp function call are not provided.
 */
#define NOMP_USER_INPUT_NOT_PROVIDED -128
/**
 * @ingroup nomp_errors
 * @brief One of the inputs to a libnomp function call are not valid.
 */
#define NOMP_USER_INPUT_IS_INVALID -129
/**
 * @ingroup nomp_errors
 * @brief Map pointer provided to libnomp is not valid.
 */
#define NOMP_USER_MAP_PTR_IS_INVALID -130
/**
 * @ingroup nomp_errors
 * @brief Map operation provided to libnomp is not applicable.
 */
#define NOMP_USER_MAP_OP_IS_INVALID -131
/**
 * @ingroup nomp_errors
 * @brief Log id provided to libnomp is not valid.
 */
#define NOMP_USER_LOG_ID_IS_INVALID -132
/**
 * @ingroup nomp_errors
 * @brief User callback function failed during execution.
 */
#define NOMP_USER_CALLBACK_FAILURE -133
/**
 * @ingroup nomp_errors
 * @brief Kernel argument type provided to libnomp is not valid.
 */
#define NOMP_USER_KNL_ARG_TYPE_IS_INVALID -134
/**
 * @ingroup nomp_errors
 * @brief User argument provided to libnomp is not valid.
 */
#define NOMP_USER_ARG_IS_INVALID -135

/**
 * @ingroup nomp_errors
 * @brief libnomp is already initialized.
 */
#define NOMP_INITIALIZE_FAILURE -256
/**
 * @ingroup nomp_errors
 * @brief Failed to finalize libnomp.
 */
#define NOMP_FINALIZE_FAILURE -258
/**
 * @ingroup nomp_errors
 * @brief One of the inputs to a libnomp function call are not valid.
 */
#define NOMP_NULL_INPUT_ENCOUNTERED -260

/**
 * @ingroup nomp_errors
 * @brief A python call made by libnomp failed.
 */
#define NOMP_PY_CALL_FAILURE -385
/**
 * @ingroup nomp_errors
 * @brief Loopy conversion failed.
 */
#define NOMP_LOOPY_CONVERSION_FAILURE -387
/**
 * @ingroup nomp_errors
 * @brief Failed to find loopy kernel.
 */
#define NOMP_LOOPY_KNL_NAME_NOT_FOUND -388
/**
 * @ingroup nomp_errors
 * @brief Code generation from loopy kernel failed.
 */
#define NOMP_LOOPY_CODEGEN_FAILURE -389
/**
 * @ingroup nomp_errors
 * @brief Code generation from loopy kernel failed.
 */
#define NOMP_LOOPY_GET_GRIDSIZE_FAILURE -390
/**
 * @ingroup nomp_errors
 * @brief Grid size calculation failed.
 */
#define NOMP_LOOPY_EVAL_GRIDSIZE_FAILURE -391

/**
 * @ingroup nomp_errors
 * @brief libnomp Cuda operation failed.
 */
#define NOMP_CUDA_FAILURE -512
/**
 * @ingroup nomp_errors
 * @brief libnomp HIP operation failed.
 */
#define NOMP_HIP_FAILURE -145
/**
 * @ingroup nomp_errors
 * @brief libnomp OpenCL failure.
 */
#define NOMP_OPENCL_FAILURE -513
/**
 * @ingroup nomp_errors
 * @brief libnomp SYCL failure.
 */
#define NOMP_SYCL_FAILURE -514
/**
 * @ingroup nomp_errors
 * @brief libnomp ISPC failure.
 */
#define NOMP_ISPC_FAILURE -515
/**
 * @ingroup nomp_errors
 * @brief libnomp source compilation failure.
 */
#define NOMP_JIT_FAILURE -524

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
 * @brief Initializes libnomp with the specified backend, platform, device, etc.
 *
 * @details Initializes nomp code generation for the specified backend (e.g.,
 * OpenCL, Cuda, etc) using command line arguments. Also platform id, device id,
 * verbose level, annotation script and annotation function can be specified as
 * well. Returns a negative value if an error occurs during the initialization,
 * otherwise returns 0. Calling this method twice (without nomp_finalize in
 * between) will return an error (but not segfault). Currently only supports
 * Cuda and OpenCL backends.
 *
 * <b>Accepted arguments:</b>
 * \arg `-b|--backend <backend-name>` Specify backend type (Default: opencl).
 * \arg `-p|--platform <platform-index>` Specify platform id (Default: 0).
 * \arg `-d|--device <device-index>` Specify device id (Default: 0).
 * \arg `-v|--verbose <verbose-level>` Specify verbose level (Default: 0).
 * \arg `-as|--annts-script <annotation-script>` Specify the directory which the
 * annotation script resides. \arg `-af|--annts-func <annotation-function>`
 * Specify the annotation function name.
 *
 * @param[in] argc The number of arguments to nomp_init().
 * @param[in] argv Arguments as strings, values followed by options.
 * @return int
 *
 * <b>Example usage:</b>
 * @code{.c}
 * const char *argv[] = {"--backend", "opencl", "-device", "0", "--platform",
 * "0"};
 * int argc = 6;
 * int err = nomp_init(argc, argv);
 * @endcode
 */
int nomp_init(int argc, const char **argv);

/**
 * @ingroup nomp_user_api
 * @brief Performs device to host (D2H) and host to device (H2D) memory
 * transfers, allocating and freeing of memory in the device.
 *
 * @param[in] ptr Pointer to the host memory location.
 * @param[in] start_idx Start index in the vector to start copying.
 * @param[in] end_idx End index in the vector to end the copying.
 * @param[in] unit_size Size of a single vector element.
 * @param[in] op Operation to perform (One of @ref nomp_update_op).
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
 * data can be passed using the \p clauses as well. After \p clauses, number of
 * arguments to the kernel must be provided. Then for each argument, three
 * values has to be passed. First is the argument name as a string. Second is
 * is the `sizeof` argument and the third if argument type (one of @ref
 * nomp_types).
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
 * const char *clauses[4] = {"transform", "file", "function", 0};
 * int err = nomp_jit(&id, knl, clauses, 3, "a", sizeof(a), NOMP_PTR, "b",
 * sizeof(b), NOMP_PTR, "N", sizeof(int), NOMP_INT);
 * @endcode
 *
 * @param[out] id Id of the generated kernel.
 * @param[in] src Kernel source in C.
 * @param[in] clauses Clauses to provide meta information about the kernel.
 * @param[in] narg Number of arguments to the kernel.
 * @param[in] ... Three values for each argument: identifier, sizeof(argument)
 * and argument type.
 * @return int
 */
int nomp_jit(int *id, const char *src, const char **clauses, int narg, ...);

/**
 * @ingroup nomp_user_api
 * @brief Runs the kernel generated by nomp_jit().
 *
 * @details Runs the kernel with a given kernel id. Kernel id is followed by the
 * arguments (i.e., pointers and pointer to scalar variables).
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
 * const char *clauses[4] = {"transform", "file", "function", 0};
 * int err = nomp_jit(&id, knl, clauses, 3, "a", sizeof(a), NOMP_PTR, "b",
 * sizeof(b), NOMP_PTR, "N", sizeof(int), NOMP_INT);
 * err = nomp_run(id, 3, a, b, N);
 * @endcode
 *
 * @param[in] id Id of the kernel to be run.
 * @param[in] ...  Arguments to the kernel.
 *
 * @return int
 */
int nomp_run(int id, ...);

/**
 * @ingroup nomp_user_api
 * @brief Synchronize task execution on device.
 *
 * Implement a host-side barrier till the device finish executing all the
 * previous nomp kernels and/or memory copies.
 *
 * @return int
 */
int nomp_sync();

/**
 * @ingroup nomp_user_api
 * @brief Check nomp API return values for errors.
 *
 * @param[in] err Return value from nomp API.
 *
 */
#define nomp_check(err)                                                        \
  {                                                                            \
    int err_ = (err);                                                          \
    if (nomp_get_log_type(err_) == NOMP_ERROR)                                 \
      return err_;                                                             \
  }

/**
 * @ingroup nomp_log_type
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
 * @brief Return the log description given the log id.
 *
 * @details Returns the log description given the log id. Returns NULL if the
 * id is invalid.
 * @param[in] id id of the error.
 * @return char*
 */
char *nomp_get_log_str(int id);

/**
 * @ingroup nomp_user_api
 * @brief Return log number.
 *
 * @details Returns the log number given the log_id. If log_id
 * is invalid return NOMP_USER_LOG_ID_IS_INVALID.
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
 * Calling this method before nomp_init() will return an error. Calling this
 * method twice will also return an error.
 *
 * @return int
 */
int nomp_finalize(void);

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_H_
