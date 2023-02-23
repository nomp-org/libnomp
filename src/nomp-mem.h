#if !defined(_NOMP_MEM_H_)
#define _NOMP_MEM_H_

#include <stddef.h>
#include <stdlib.h>

/**
 * @defgroup nomp_mem_utils Host memory management functions
 */

/**
 * @ingroup nomp_mem_utils
 * @brief Deallocates or frees a memory block.
 *
 * @param p Pointer to the memory to deallocate.
 * @param file Name of the file in which the function is called.
 * @param line Line number in the file where the function is called.
 * @return int
 */
int nomp_free_(void *p, const char *file, unsigned line);

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for deallocating or freeing a memory block using
 * nomp_free_(). File name and line number are passed implicitly.
 *
 * @param x Pointer to the memory to deallocate.
 * @return int
 */
#define nomp_free(x) nomp_free_(x, __FILE__, __LINE__)

/**
 * @ingroup nomp_mem_utils
 * @brief Allocates memory blocks.
 *
 * @param size Bytes to allocate.
 * @param file Name of the file in which the function is called.
 * @param line Line number in the file where the function is called.
 * @return Void pointer
 */
void *nomp_malloc_(size_t size, const char *file, unsigned line);

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for allocating memory blocks using nomp_malloc_().
 * File name and line number are passed implicitly.
 *
 * @param T Type of element.
 * @param count Number of elements.
 * @return Pointer of type T.
 */
#define nomp_malloc(T, count)                                                  \
  ((T *)nomp_malloc_((count) * sizeof(T), __FILE__, __LINE__))

/**
 * @ingroup nomp_mem_utils
 * @brief Allocates an array in memory with elements initialized to 0.
 *
 * @param nmemb Number of elements.
 * @param size Length in bytes of each element.
 * @param file Name of the file in which the function is called.
 * @param line Line number in the file where the function is called.
 * @return Void pointer
 */
void *nomp_calloc_(size_t nmemb, size_t size, const char *file, unsigned line);

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

/**
 * @ingroup nomp_mem_utils
 * @brief Reallocate memory blocks.
 *
 * @param ptr Pointer to the memory area to be reallocated.
 * @param size New size in bytes.
 * @param file Name of the file in which the function is called.
 * @param line Line number in the file where the function is called.
 * @return Void pointer
 */
void *nomp_realloc_(void *ptr, size_t size, const char *file, unsigned line);

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

#endif // _NOMP_MEM_H_
