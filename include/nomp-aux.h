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
 *
 * @brief Various helper functions used for implementing the library.
 */

/**
 * @ingroup nomp_other_utils
 *
 * @brief Concatenates \p n strings.
 *
 * Concatenates \p n strings and returns a pointer to the resulting string.
 * Each string should be at most \p max_len length long. User has to free memory
 * allocated for the resulting string using nomp_free().
 *
 * @param[in] n Number of strings to concatenate.
 * @param[in] max_len Maximum length of an individual string.
 * @param[in] ... Strings to concatenate.
 * @return char*
 */
char *nomp_str_cat(unsigned n, unsigned max_len, ...);

/**
 * @ingroup nomp_other_utils
 *
 * @brief Try to convert input string \p str to an non-negative integer value.
 *
 * Convert input string \p str to an non-negative int value. Returns converted
 * non-negative int value if successful, otherwise return -1. \p size denotes
 * the maximum length of the string \p str.
 *
 * @param[in] str String to convert into unsigned int.
 * @param[in] size Length of the string.
 * @return int
 */
int nomp_str_toui(const char *str, size_t size);

/**
 * @ingroup nomp_other_utils
 *
 * @brief Returns the maximum among all integers passed.
 *
 * Returns the maximum between \p n integers.
 *
 * @param[in] n Total number of integers.
 * @param[in] ... List of integers.
 * @return int
 */
int nomp_max(unsigned n, ...);

/**
 * @ingroup nomp_other_utils
 *
 * @brief Get a copy of the environment variable value if it exists.
 *
 * Get a copy of the value of the environment variable \p name as a string if
 * it is defined. Return NULL if the environment variable is not set. \p size
 * is the maximum length of the string. User must free the memory allocated for
 * the resulting string using nomp_free().
 *
 * @param[in] name Name of the environment variable.
 * @param[in] size Maximum length of the environment variable value.
 * @return char *
 */
char *nomp_copy_env(const char *name, size_t size);

/**
 * @ingroup nomp_other_utils
 *
 * @brief Returns the length of a posix complaint path.
 *
 * If \p len is not NULL, it is set to the length of the \p path if the \p path
 * length resolution was successful. Otherwise, it is set to zero. On success,
 * the function returns zero. Otherwise, it returns an error id which can be
 * used to query the error id and string using nomp_get_err_str() and
 * nomp_get_err_id().
 *
 * @param[out] len Lenth of path specified in \p path.
 * @param[in] path Path to get the maximum length.
 * @return int
 */
int nomp_path_len(size_t *len, const char *path);

/**
 * @ingroup nomp_other_utils
 *
 * @brief Check if the python script path exists.
 *
 * Check if there is a python script at the path \p path. Returns 0 if the
 * path exists, otherwise returns an error id which can be used to query the
 * error id and string using nomp_get_err_str() and nomp_get_err_id(). The \p
 * path should be provided without the ".py" extension.
 *
 * @param[in] path Path to the python script without the ".py" extension.
 * @return int
 */
int nomp_check_py_script_path(const char *path);

#ifdef __cplusplus
}
#endif

#endif // _NOMP_AUX_H_
