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

/**
 * @ingroup nomp_mem_utils
 * @brief Deallocates or frees a memory block.
 *
 * @param p Pointer to the memory to deallocate.
 * @param file Name of the file in which the function is called.
 * @param line Line number in the file where the function is called.
 * @return int
 */
int sfree(void *p, const char *file, unsigned line);

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for deallocating or freeing a memory block using sfree().
 * File name and line number are passed implicitly.
 *
 * @param x Pointer to the memory to deallocate.
 * @return int
 */
#define tfree(x) sfree(x, __FILE__, __LINE__)

/**
 * @ingroup nomp_mem_utils
 * @brief Allocates memory blocks.
 *
 * @param size Bytes to allocate.
 * @param file Name of the file in which the function is called.
 * @param line Line number in the file where the function is called.
 * @return Void pointer
 */
void *smalloc(size_t size, const char *file, unsigned line);

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for allocating memory blocks using smalloc().
 * File name and line number are passed implicitly.
 *
 * @param T Type of element.
 * @param count Number of elements.
 * @return Pointer of type T.
 */
#define tmalloc(T, count)                                                      \
  ((T *)smalloc((count) * sizeof(T), __FILE__, __LINE__))

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
void *scalloc(size_t nmemb, size_t size, const char *file, unsigned line);

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for allocating an array in memory with elements
 * initialized to 0 using scalloc(). File name and line number are passed
 * implicitly.
 *
 * @param T Type of element.
 * @param count Number of elements.
 * @return Pointer of type T.
 */
#define tcalloc(T, count) ((T *)scalloc((count), sizeof(T), __FILE__, __LINE__))

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
void *srealloc(void *ptr, size_t size, const char *file, unsigned line);

/**
 * @ingroup nomp_mem_utils
 * @brief Helper macro for reallocating memory blocks using srealloc().
 * File name and line number are passed implicitly.
 *
 * @param ptr Pointer to the memory area to be reallocated.
 * @param T Type of element.
 * @param count Number of elements.
 * @return Pointer of type T
 */
#define trealloc(ptr, T, count)                                                \
  ((T *)srealloc((ptr), (count) * sizeof(T), __FILE__, __LINE__))

#endif // _NOMP_MEM_H_
