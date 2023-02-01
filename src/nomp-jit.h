#if !defined(_NOMP_JIT_H_)
#define _NOMP_JIT_H_

/**
 * @defgroup nomp_compile_utils Functions to compile source at runtime.
 */

/**
 * @ingroup nomp_compile_utils
 * @brief JIT compile a source string at runtime.
 *
 * JIT Compile a source string at runtime using a specified compiler, flags and
 * a working directory. \p id is set to dynamically loaded \p entry point in the
 * JIT compiled program. \p id should be set to -1 on input and is set to a
 * non-negative value upon successful exit. On success, jit_compile() returns 0
 * and a positive value otherwise.
 *
 * @param[out] id Handle to the \p entry in the compiled binary file.
 * @param[in] source Source to be compiled at runtime.
 * @param[in] cc Full path to the compiler.
 * @param[in] cflags Compile flags to be used during compilation.
 * @param[in] entry Entry point (usually the name of function to be called) to
 * the source.
 * @param[in] wrkdir Working directory to generate outputs and store
 * temporaries.
 *
 * @return int
 */
int jit_compile(int *id, const char *source, const char *cc, const char *cflags,
                const char *entry, const char *wrkdir);

/**
 * @ingroup nomp_compile_utils
 * @brief Run a JIT compiled program.
 *
 * @param[in] id Handle of the JIT compiled program.
 * @param[in] n Number of arguments to the program.
 * @param[in] ... Variable list of arguments to the program.
 *
 * @return int
 */
int jit_run(int id, int n, ...);

/**
 * @ingroup nomp_compile_utils
 * @brief Free a jit compiled program.
 *
 * Free a jit compiled program. On successful exit, \p id is set to -1.
 *
 * @param[in] id Handle of the JIT compiled program.
 *
 * @return int
 */
int jit_free(int *id);

#endif // _NOMP_JIT_H_
