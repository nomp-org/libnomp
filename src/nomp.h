#if !defined(_LIB_NOMP_H_)
#define _LIB_NOMP_H_

#include <stddef.h>

/**
 * @defgroup nomp_map_direction Map Direction
 * Defines the operation direction in `nomp_map`.
 * TODO: Probably these should be an enum.
 */

/**
 * @ingroup nomp_map_direction
 * @brief NOMP allocation operation.
 */
#define NOMP_ALLOC 1
/**
 * @ingroup nomp_map_direction
 * @brief Mapping of host to device(H2D) operation.
 */
#define NOMP_H2D 2
/**
 * @ingroup nomp_map_direction
 * @brief Mapping of device to host(D2H) operation.
 */
#define NOMP_D2H 4
/**
 * @ingroup nomp_map_direction
 * @brief NOMP freeing operation.
 */
#define NOMP_FREE 8

/**
 * @defgroup nomp_types Types
 * Defines argument type.
 * TODO: Probably these should be an enum.
 */

/**
 * @ingroup nomp_types
 * @brief NOMP integer type
 */
#define NOMP_INTEGER 1
/**
 * @ingroup nomp_types
 * @brief NOMP float type
 */
#define NOMP_FLOAT 2
/**
 * @ingroup nomp_types
 * @brief NOMP pointer type
 */
#define NOMP_PTR 4

/**
 * @defgroup nomp_errors Errors
 * TODO: Probably these should be an enum.
 */

/**
 * @ingroup nomp_errors
 * @brief Invalid NOMP backend
 */
#define NOMP_INVALID_BACKEND -32
/**
 * @ingroup nomp_errors
 * @brief Invalid NOMP platform
 */
#define NOMP_INVALID_PLATFORM -33
/**
 * @ingroup nomp_errors
 * @brief Invalid NOMP device
 */
#define NOMP_INVALID_DEVICE -34
/**
 * @ingroup nomp_errors
 * @brief Invalid NOMP map pointer
 */
#define NOMP_INVALID_MAP_PTR -36
/**
 * @ingroup nomp_errors
 * @brief Invalid NOMP map operation
 */
#define NOMP_INVALID_MAP_OP -37
/**
 * @ingroup nomp_errors
 * @brief Pointer is already mapped
 */
#define NOMP_PTR_ALREADY_MAPPED -38
/**
 * @ingroup nomp_errors
 * @brief Invalid NOMP kernel
 */
#define NOMP_INVALID_KNL -39

/**
 * @ingroup nomp_errors
 * @brief NOMP is already initialized
 */
#define NOMP_INITIALIZED_ERROR -64
/**
 * @ingroup nomp_errors
 * @brief NOMP is not initialized
 */
#define NOMP_NOT_INITIALIZED_ERROR -65
/**
 * @ingroup nomp_errors
 * @brief Failed to finalize NOMP
 */
#define NOMP_FINALIZE_ERROR -66
/**
 * @ingroup nomp_errors
 * @brief NOMP malloc error
 */
#define NOMP_MALLOC_ERROR -67

/**
 * @ingroup nomp_errors
 * @brief NOMP python initialization failed
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
#define NOMP_LOOPY_GRIDSIZE_FAILED -103
/**
 * @ingroup nomp_errors
 * @brief Grid size calculation failed
 */
#define NOMP_GRIDSIZE_CALCULATION_FAILED -103

/**
 * @ingroup nomp_errors
 * @brief NOMP kernel build failed
 */
#define NOMP_KNL_BUILD_ERROR -128
/**
 * @ingroup nomp_errors
 * @brief Invalid NOMP kernel argument type
 */
#define NOMP_KNL_ARG_TYPE_ERROR -129
/**
 * @ingroup nomp_errors
 * @brief Setting NOMP kernel argument failed
 */
#define NOMP_KNL_ARG_SET_ERROR -130
/**
 * @ingroup nomp_errors
 * @brief NOMP kernel run failed
 */
#define NOMP_KNL_RUN_ERROR -131

#define NOMP_OUT_OF_MEMORY -140
#define NOMP_INVALID_ERROR_ID -141

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_user_api API
 */

/**
 * @ingroup nomp_user_api
 * @brief Initializes libnomp with the specified backend, platform and device.
 *
 * @details Initializes the nomp code generation for the specified backend
 * (e.g., Cuda, OpenCL, etc) using the given platform id and device id. Returns
 * a negative value if an error occurs during the initialization, otherwise
 * returns 0. Calling this method twice (without nomp_finalize in between) will
 * return an error (but not segfault) as well. Currently only supports Cuda and
 * OpenCL backends.
 *
 * @param[in] backend Target backend for code generation ("Cuda", "OpenCL",
 * etc.).
 * @param[in] platform Target platform id to share resources and execute kernels
 *                 in the targeted device (only used for OpenCL backend).
 * @param[in] device Target device id to execute kernels.
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
 * transfers, allocation and freeing of memory on the device.
 *
 * @param[in] ptr Pointer to the host vector.
 * @param[in] start_idx Start index in the vector.
 * @param[in] end_idx End index in the vector.
 * @param[in] unit_size Size of a single vector element.
 * @param[in] op Operation to perform (One of @ref nomp_map_direction).
 * @return int
 *
 * @details Operation op will be performed on the array slice [start_idx,
 * end_idx), i.e., on array elements start_idx, ... end_idx - 1. This method
 * returns a non-zero value if there is an error and 0 otherwise.
 *
 * <b>Example usage:</b>
 * @code{.c}
 * const int N = 10;
 * double a[10], b[10];
 * for (unsigned i = 0; i < N; i++) {
 *   a[i] = i;
 *   b[i] = N - i;
 * }
 * int err = nomp_map(a, 0, N, sizeof(double), NOMP_H2D);
 * int err = nomp_map(b, 0, N, sizeof(double), NOMP_H2D);
 * // Code that change array values on the device (e.g., execution of a kernel)
 * int err = nomp_map(a, 0, N, sizeof(double), NOMP_D2H);
 * int err = nomp_map(b, 0, N, sizeof(double), NOMP_D2H);
 * int err = nomp_map(a, 0, N, sizeof(double), NOMP_FREE);
 * int err = nomp_map(b, 0, N, sizeof(double), NOMP_FREE);
 * @endcode
 */
int nomp_map(void *ptr, size_t start_idx, size_t end_idx, size_t unit_size,
             int op);

/**
 * @ingroup nomp_user_api
 * @brief Generate a kernel for the target backend (OpenCL, Cuda, etc.) from C
 * source.
 *
 * @details User defined code transformations will be applied based on the
 * annotations passed to this function. Kernel specific transformations can be
 * specified in the callback function.
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
 * int err = nomp_jit(&id, &ndim, global, local, knl, NULL, "file:function",
 *                    3, "a,b,N", NOMP_PTR, sizeof(double), a, NOMP_PTR,
 *                    sizeof(double), b, NOMP_INTEGER, sizeof(int), &N);
 * @endcode
 *
 * @param[out] id id of the generated kernel.
 * @param[out] ndim Number of dimensions of the kernel (This param will be
 * removed).
 * @param[out] global Global grid (This param will be removed).
 * @param[out] local Local grid (This param will be removed).
 * @param[in] c_src Kernel source in C.
 * @param[in] annotations Annotations to perform user defined (domain specific)
 * transformations.
 * @param[in] callback Callback function that is called when generating the
 * kernel.
 * @param[in] nargs Number of arguments to the kernel.
 * @param[in] args Comma separated list of argument names.
 * @param[in] ... For each argument, we pass the argument type (one of @ref
 * nomp_types), size of the base type and the pointer to the argument.
 *
 * @return int
 */
int nomp_jit(int *id, int *ndim, size_t *global, size_t *local,
             const char *c_src, const char *annotations, const char *callback,
             int nargs, const char *args, ...);

/**
 * @ingroup nomp_user_api
 * @brief Runs the kernel.
 *
 * @details Runs the kernel with a given kernel id. Kernel id is followed by the
 * number of arguments. Then for each argument we pass the argument type (@ref
 * nomp_type) size of the base type in case of an integer and pointer to the
 * argument.
 *
 * @param[in] id id of the kernel to be run
 * @param[in] ndim Number of dimensions of the kernel (This param will be
 * removed)
 * @param[in] global Global grid (This param will be removed)
 * @param[in] local Local grid (This param will be removed)
 * @param[in] nargs Number of arguments
 * @param[in] ... For each argument, argument type, sizeof base type and pointer
 * to the argument.
 *
 * @return int
 */
int nomp_run(int id, int ndim, const size_t *global, const size_t *local,
             int nargs, ...);

void nomp_assert_(int cond, const char *file, unsigned line);
#define nomp_assert(cond) nomp_assert_(cond, __FILE__, __LINE__)

void nomp_chk_(int err, const char *file, unsigned line);
#define nomp_chk(err) nomp_chk_(err, __FILE__, __LINE__)

int nomp_set_error_(const char *description, int type, const char *file_name,
                    unsigned line_no);
#define nomp_set_error(description, type)                                      \
  nomp_set_error_(description, type, __FILE__, __LINE__);

int nomp_get_error(char **error, int error_id);

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
