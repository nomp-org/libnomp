#if !defined(_NOMP_AUX_H_)
#define _NOMP_AUX_H_

#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * @defgroup nomp_other_utils Other helper functions.
 */

/**
 * @ingroup nomp_other_utils
 * @brief Concatenates \p n strings.
 *
 * Concatenates \p n strings and returns a pointer to the resulting string.
 * Each string should be at most \p max_len length long.
 *
 * @param[in] n Number of strings to concatenate.
 * @param[in] max_len Maximum length of an individual string.
 * @param[in] ... Strings to concatenate.
 * @return char*
 */
char *strcatn(unsigned n, unsigned max_len, ...);

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
int strntoui(const char *str, size_t size);

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
int pathlen(size_t *len, const char *path);

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
int maxn(unsigned n, ...);

/**
 * @ingroup nomp_other_utils
 * @brief Returns a non-zero error if the input is NULL.
 *
 * This function call nomp_set_log() to register an error if the input is NULL.
 * Use the macro nomp_null_input() to automatically add last three arguments.
 *
 * @param[in] p Input pointer.
 * @param[in] func Function in which the null check is done.
 * @param[in] line Line number where the null check is done.
 * @param[in] file File name in which the null check is done.
 * @return int
 */
int check_null_input_(void *p, const char *func, unsigned line,
                      const char *file);
#define check_null_input(p)                                                    \
  nomp_check(check_null_input_((void *)(p), __func__, __LINE__, __FILE__))

#endif // _NOMP_AUX_H_
