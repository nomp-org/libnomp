#if !defined(_NOMP_MEM_H_)
#define _NOMP_MEM_H_

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_mem_utils Host memory management functions
 */

inline static void *check_if_null(void *temp, const char *file,
                                  unsigned int line,
                                  const char  *function_name) {
  if (temp == NULL) {
    fprintf(stderr, "[Error] %s:%u: Failed to allocate memory with nomp_%s\n",
            file, line, function_name);
    exit(EXIT_FAILURE);
  }
  return temp;
}

inline static void nomp_free_(void **p) { free(*p), *p = NULL; }

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for deallocating or freeing a memory block using
 * nomp_free_(). File name and line number are passed implicitly.
 *
 * @param p Address of the pointer to the memory to deallocate.
 * @return void
 */
#define nomp_free(p) nomp_free_((void **)(p))

inline static void *nomp_calloc_(size_t count, size_t size, const char *file,
                                 unsigned line) {
  void *temp = calloc(count, size);
  return check_if_null(temp, file, line, "calloc");
}

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for allocating an array in memory with elements
 * initialized to 0 using nomp_calloc_(). File name and line number are passed
 * implicitly.
 *
 * @param T Type of element.
 * @param count Number of elements.
 * @return Pointer of type T.
 */
#define nomp_calloc(T, count)                                                  \
  ((T *)nomp_calloc_((count), sizeof(T), __FILE__, __LINE__))

inline static void *nomp_realloc_(void *ptr, size_t size, const char *file,
                                  unsigned line) {
  void *temp = realloc(ptr, size);
  return check_if_null(temp, file, line, "realloc");
}

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for reallocating memory blocks using nomp_realloc_().
 * File name and line number are passed implicitly.
 *
 * @param ptr Pointer to the memory area to be reallocated.
 * @param T Type of element.
 * @param count Number of elements.
 * @return Pointer of type T
 */
#define nomp_realloc(ptr, T, count)                                            \
  ((T *)nomp_realloc_((ptr), (count) * sizeof(T), __FILE__, __LINE__))

#ifdef __cplusplus
}
#endif

#endif // _NOMP_MEM_H_
