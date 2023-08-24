#if !defined(_NOMP_JIT_H_)
#define _NOMP_JIT_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_compile_utils JIT compile utilities
 * @brief Functions to JIT compile source files at runtime.
 */

int nomp_jit_compile(int *id, const char *source, const char *cc,
                     const char *cflags, const char *entry, const char *wrkdir,
                     const char *srcf, const char *libf);

int nomp_jit_run(int id, void *p[]);

int nomp_jit_free(int *id);

#ifdef __cplusplus
}
#endif

#endif // _NOMP_JIT_H_
