#if !defined(_LIB_NOMP_H_)
#define _LIB_NOMP_H_

#include <stddef.h>

// Map Direction: Probably should be an enum
#define NOMP_ALLOC 1
#define NOMP_H2D 2
#define NOMP_D2H 4
#define NOMP_FREE 8

// Types: Probably these should be an enum
#define NOMP_SCALAR 0
#define NOMP_PTR 1

// Errors
#define NOMP_INVALID_BACKEND -32
#define NOMP_INVALID_PLATFORM -33
#define NOMP_INVALID_DEVICE -34
#define NOMP_INVALID_MAP_PTR -36
#define NOMP_INVALID_MAP_OP -37
#define NOMP_INVALID_KNL -38

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

int nomp_init(const char *backend, int platform, int device);

int nomp_map(void *ptr, size_t start_idx, size_t end_idx, size_t unit_size,
             int op);

int nomp_jit(int *id, int *ndim, size_t *global, size_t *local,
             const char *c_src, const char *annotations, const char *callback,
             int nargs, const char *args, ...);

int nomp_run(int id, int ndim, const size_t *global, const size_t *local,
             int nargs, ...);

int nomp_err(char *buf, int err, size_t buf_size);

int nomp_finalize(void);

void nomp_chk_(int err, const char *file, unsigned line);
#define nomp_chk(err) nomp_chk_(err, __FILE__, __LINE__)

void nomp_assert_(int cond, const char *file, unsigned line);
#define nomp_assert(cond) nomp_assert_(cond, __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif // _LIB_NOMP_H_
