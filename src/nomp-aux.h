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
 * @brief Concatenates atmost `nstr` strings.
 *
 * Concatenates atmost `nstr` strings and returns a pointer to
 * resulting string.
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
 * Convert input string `str` to an unsigned int value. Returns converted
 * unsigned int value if successful, otherwise return -1.
 *
 * @param[in] str String to convert into unsigned int.
 * @param[in] size Length of the string.
 * @return int
 */
int strntoui(const char *str, size_t size);

/**
 * @ingroup nomp_other_utils
 * @brief Returns maximum length of a path.
 *
 * Returns the maximum length of specified path.
 *
 * @param[in] len Lenth of path specified in \p path.
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
 * This function call set_log() to register an error if the input is NULL.
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
  return_on_err(check_null_input_((void *)(p), __func__, __LINE__, __FILE__))

#endif // _NOMP_AUX_H_
