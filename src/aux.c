#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "nomp-aux.h"
#include "nomp-log.h"
#include "nomp-mem.h"

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
char *nomp_str_cat(unsigned n, unsigned max_len, ...) {
  va_list vargs;
  va_start(vargs, max_len);

  unsigned max = NOMP_MAX_BUFFER_SIZE, len = 0;
  char *out = nomp_calloc(char, max);
  for (unsigned i = 0; i < n; i++) {
    const char *s = va_arg(vargs, const char *);
    if (max <= len + strnlen(s, max_len)) {
      max = 3 * (len + strnlen(s, max_len)) / 2 + 1;
      out = nomp_realloc(out, char, max);
    }
    strncpy(out + len, s, strnlen(s, max_len));
    len += strnlen(s, max_len);
  }
  out[len] = '\0';

  va_end(vargs);

  return out;
}

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
int nomp_str_toui(const char *str, size_t size) {
  if (str == NULL)
    return -1;

  char *copy = strndup(str, size), *end_ptr;
  int num = (int)strtol(copy, &end_ptr, 10);
  if (copy == end_ptr || *end_ptr != '\0' || num < 0)
    num = -1;
  nomp_free(&copy);

  return num;
}

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
int nomp_max(unsigned n, ...) {
  va_list valist;
  va_start(valist, n);

  int max = INT_MIN;
  for (unsigned i = 0; i < n; i++) {
    int cur = va_arg(valist, int);
    if (max < cur)
      max = cur;
  }

  va_end(valist);

  return max;
}

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
char *nomp_copy_env(const char *name, size_t size) {
  const char *tmp = getenv(name);
  if (tmp)
    return strndup(tmp, size);
  return NULL;
}

/**
 * @ingroup nomp_other_utils
 *
 * @brief Returns the length of a posix complaint path.
 *
 * If \p len is not NULL, it is set to the length of the \p path if the \p path
 * length resolution was successful. Otherwise, it is set to zero. On success,
 * the function returns zero. Otherwise, it returns an error id which can be
 * used to query the error id and string using nomp_get_err_str() and
 * nomp_get_err_no().
 *
 * @param[out] len Lenth of path specified in \p path.
 * @param[in] path Path to get the maximum length.
 * @return int
 */
int nomp_path_len(size_t *len, const char *path) {
  if (len)
    *len = 0;
  char *abs = realpath(path, NULL);
  if (!abs) {
    return nomp_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                    "Unable to find path: \"%s\". Error: %s.", path,
                    strerror(errno));
  }

  if (len)
    *len = strnlen(abs, PATH_MAX);
  nomp_free(&abs);

  return 0;
}
