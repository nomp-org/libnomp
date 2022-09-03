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
 * @brief Invalid NOMP kernal
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
 * @brief Failed to find loopy kernal
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
 * Initializes the nomp code generation for the specified backend (e.g., Cuda,
 * OpenCL, etc) using the given platform id and device id. Returns a negative
 * value if an error occurs during the initialization, otherwise returns 0.
 * Calling this method twice (without nomp_finalize in between) will return an
 * error as well. Currently only supports Cuda and OpenCL backends.
 *
 * @param backend Target backend for code generation.
 * @param platform Target platform id to share resources and execute kernals
 *                 in the targeted device.
 * @param device Target device id to execute kernals.
 * @return int
 */
int nomp_init(const char *backend, int platform, int device);

/**
 * @ingroup nomp_user_api
 * @brief Does D2H/H2D transfers (update) and alloc/free.
 *
 * Does data tramsfers from host to device and device to host, allocations and
 * freeing of memory where the operation is specified by `op`.
 *
 * @param ptr Pointer to the vector
 * @param start_idx Starting index
 * @param end_idx End index
 * @param unit_size Unit size of a vector element
 * @param op Operation to perform
 * @return int
 */
int nomp_map(void *ptr, size_t start_idx, size_t end_idx, size_t unit_size,
             int op);

/**
 * @ingroup nomp_user_api
 * @brief JIT compile the kernels after applying code transformations.
 *
 * <b>Example</b>
 * @code{.c}
 * int err = nomp_jit(&id, &ndim, global, local, knl, NULL, "file:function",
 *                    3, "a,b,N", NOMP_PTR, sizeof(double), a, NOMP_PTR,
 *                    sizeof(double), b, NOMP_INTEGER, sizeof(int), &N);
 * @endcode
 *
 * @param id Kernal id
 * @param ndim Number of dimensions of the kernel
 * @param global Global grid
 * @param local Local grid
 * @param c_src Kernal
 * @param annotations Annotations
 * @param callback Callbacks
 * @param nargs Number of arguments
 * @param args Comma separated arguments
 * @param ...
 *
 * @return int
 */
int nomp_jit(int *id, int *ndim, size_t *global, size_t *local,
             const char *c_src, const char *annotations, const char *callback,
             int nargs, const char *args, ...);

/**
 * @ingroup nomp_user_api
 * @brief Runs the kernels.
 *
 * @param id Kernal id
 * @param ndim Number of dimensions of the kernel
 * @param global Global grid
 * @param local Local grid
 * @param nargs Number of arguments
 * @param ...
 * @return int
 */
int nomp_run(int id, int ndim, const size_t *global, const size_t *local,
             int nargs, ...);

/**
 * @ingroup nomp_user_api
 * @brief Copies nomp error to the specified buffer.
 *
 * @param buf Buffer to copy error description
 * @param err Nomp error
 * @param buf_size Buffer size
 * @return int
 */
int nomp_err(char *buf, int err, size_t buf_size);

/**
 * @ingroup nomp_user_api
 * @brief Finalizes libnomp.
 *
 * Frees allocated runtime resources for libnomp. Returns a
 * negative value if an error occurs during the finalize
 * process, otherwise returns 0. Calling this method before
 * `nomp_init` will retrun an error. Calling this method twice
 * will also return an error.
 *
 * @return int
 */
int nomp_finalize(void);

void nomp_chk_(int err, const char *file, unsigned line);
#define nomp_chk(err) nomp_chk_(err, __FILE__, __LINE__)

void nomp_assert_(int cond, const char *file, unsigned line);
#define nomp_assert(cond) nomp_assert_(cond, __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_H_
