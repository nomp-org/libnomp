#if !defined(_LIB_NOMP_H_)
#define _LIB_NOMP_H_

#include <stddef.h>

// Map Direction: Probably should be an enum
#define NOMP_ALLOC 1
#define NOMP_H2D 2
#define NOMP_D2H 4
#define NOMP_FREE 8

// Types: Probably these should be an enum
#define NOMP_INTEGER 1
#define NOMP_FLOAT 2
#define NOMP_PTR 4

// Errors
/**
 * @brief Invalid NOMP backend
 */
#define NOMP_INVALID_BACKEND -32
/**
 * @brief Invalid NOMP platform
 */
#define NOMP_INVALID_PLATFORM -33
/**
 * @brief Invalid NOMP device
 */
#define NOMP_INVALID_DEVICE -34
/**
 * @brief Invalid NOMP map pointer
 */
#define NOMP_INVALID_MAP_PTR -36
/**
 * @brief Invalid NOMP map operation
 */
#define NOMP_INVALID_MAP_OP -37
/**
 * @brief Pointer is already mapped
 */
#define NOMP_PTR_ALREADY_MAPPED -38
/**
 * @brief Invalid NOMP kernal
 */
#define NOMP_INVALID_KNL -39

#define NOMP_INITIALIZED_ERROR -64
#define NOMP_NOT_INITIALIZED_ERROR -65
#define NOMP_FINALIZE_ERROR -66
#define NOMP_MALLOC_ERROR -67

#define NOMP_PY_INITIALIZE_ERROR -96
#define NOMP_INSTALL_DIR_NOT_FOUND -97
#define NOMP_USER_CALLBACK_NOT_FOUND -98
#define NOMP_USER_CALLBACK_FAILURE -99

#define NOMP_LOOPY_CONVERSION_ERROR -100
#define NOMP_LOOPY_KNL_NAME_NOT_FOUND -101
#define NOMP_LOOPY_CODEGEN_FAILED -102
#define NOMP_LOOPY_GRIDSIZE_FAILED -103
#define NOMP_GRIDSIZE_CALCULATION_FAILED -103

#define NOMP_KNL_BUILD_ERROR -128
#define NOMP_KNL_ARG_TYPE_ERROR -129
#define NOMP_KNL_ARG_SET_ERROR -130
#define NOMP_KNL_RUN_ERROR -131

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes libnomp with the specified backend.
 * 
 * Initializes the nomp code generation for the specified backend
 * using the given platform and device. Returns an error if there is
 * any error occured while the initialization otherwise 0. Calling 
 * this method twice will also return an error.
 * Currently only supports cuda and opencl backends.
 * 
 * @param backend Targeted backend for code generation.
 * @param platform Target platform id to share resources and execute kernals
 *                 in the targeted device.
 * @param device Target device id to execute kernals.
 * @return int
 */
int nomp_init(const char *backend, int platform, int device);

/**
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
 * @brief Copies nomp error to the specified buffer.
 * 
 * @param buf Buffer to copy error description
 * @param err Nomp error
 * @param buf_size Buffer size
 * @return int 
 */
int nomp_err(char *buf, int err, size_t buf_size);

/**
 * @brief Finalizes libnomp.
 * 
 * Frees allocated runtime resources for libnomp. Returns an 
 * error if there is any error occured while the finalize
 * process otherwise 0. Calling this method before `nomp_init` 
 * will retrun an error. Calling this method twice will also
 * return an error.
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
