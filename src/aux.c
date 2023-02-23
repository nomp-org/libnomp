#include "nomp-impl.h"

char *strcatn(unsigned n, unsigned max_len, ...) {
  va_list vargs;
  va_start(vargs, max_len);

  unsigned max = MAX_BUFSIZ, len = 0;
  char *out = nomp_calloc(char, max);
  for (int i = 0; i < n; i++) {
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

int strntoui(const char *str, size_t size) {
  if (str == NULL)
    return -1;

  char *copy = strndup(str, size), *end_ptr;
  int num = (int)strtol(copy, &end_ptr, 10);
  if (copy == end_ptr || *end_ptr != '\0' || num < 0)
    num = -1;
  nomp_free(copy);

  return num;
}

int pathlen(size_t *len, const char *path) {
  char *abs = realpath(path, NULL);
  if (!abs) {
    if (len)
      *len = 0;
    return set_log(NOMP_USER_INPUT_IS_INVALID, NOMP_ERROR,
                   "Unable to find path: \"%s\". Error: %s.", path,
                   strerror(errno));
  }

  if (len)
    *len = strnlen(abs, PATH_MAX);
  nomp_free(abs);

  return 0;
}

int maxn(unsigned n, ...) {
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

int check_null_input_(void *p, const char *func, unsigned line,
                      const char *file) {
  if (!p) {
    return set_log(NOMP_NULL_INPUT_ENCOUNTERED, NOMP_ERROR,
                   "Input pointer passed to function \"%s\" at line %d in file "
                   "%s is NULL.",
                   func, line, file);
  }
  return 0;
}
