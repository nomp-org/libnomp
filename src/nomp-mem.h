#if !defined(_NOMP_MEM_H_)
#define _NOMP_MEM_H_

#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_mem_utils Host memory management functions
 */

static void nomp_free_(void **p) { free(*p), *p = NULL; }

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for deallocating or freeing a memory block using
 * nomp_free_(). File name and line number are passed implicitly.
 *
 * @param p Address of the pointer to the memory to deallocate.
 * @return void
 */
#define nomp_free(p) nomp_free_((void **)p)

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for allocating memory blocks using nomp_malloc_().
 * File name and line number are passed implicitly.
 *
 * @param T Type of element.
 * @param count Number of elements.
 * @return Pointer of type T.
 */
#define nomp_malloc(T, count) ((T *)malloc((count) * sizeof(T)))

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
#define nomp_calloc(T, count) ((T *)calloc((count), sizeof(T)))

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
#define nomp_realloc(ptr, T, count) ((T *)realloc((ptr), (count) * sizeof(T)))

#ifdef __cplusplus
}
#endif

#endif // _NOMP_MEM_H_
