#if !defined(_NOMP_AUX_H_)
#define _NOMP_AUX_H_

#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nomp_other_utils Other helper functions.
 */

/**
 * @ingroup nomp_other_utils
 * @brief Concatenates \p n strings.
 *
 * Concatenates \p n strings and returns a pointer to the resulting string.
 * Each string should be at most \p max_len length long. Returned pointer
 * need to be freed using nomp_free().
 *
 * @param[in] n Number of strings to concatenate.
 * @param[in] max_len Maximum length of an individual string.
 * @param[in] ... Strings to concatenate.
 * @return char*
 */
char *nomp_str_cat(unsigned n, unsigned max_len, ...);

/**
 * @ingroup nomp_other_utils
 * @brief Convert a string to unsigned long value if possible.
 *
 * Convert input string \p str to an unsigned int value. Returns converted
 * unsigned int value if successful, otherwise return -1. \p size denotes
 * the maximum length of the string \p str.
 *
 * @param[in] str String to convert into unsigned int.
 * @param[in] size Length of the string.
 * @return int
 */
int nomp_str_toui(const char *str, size_t size);

/**
 * @ingroup nomp_other_utils
 * @brief Returns the length of a posix complaint path.
 *
 * If \p len is not NULL, it is set to the length of the path if the path length
 * resoultion was successful. Otherwise, it is set to zero.
 *
 * @param[out] len Lenth of path specified in \p path.
 * @param[in] path Path to get the maximum length.
 * @return int
 */
int nomp_path_len(size_t *len, const char *path);

/**
 * @ingroup nomp_other_utils
 * @brief Returns maximum among all integers passed.
 *
 * Returns the maximum between two or more integers.
 *
 * @param[in] n Total number of integers.
 * @param[in] ... List of integers to find the maximum of as a variable argument
 * list.
 * @return int
 */
int nomp_max(unsigned n, ...);

#ifdef __cplusplus
}
#endif

#endif // _NOMP_AUX_H_
