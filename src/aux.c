#include "nomp-impl.h"

char *strcatn(unsigned n, unsigned max_len, ...) {
  va_list vargs;
  va_start(vargs, max_len);

  unsigned max = MAX_BUFSIZ, len = 0;
  char *out = tcalloc(char, max);
  for (int i = 0; i < n; i++) {
    const char *s = va_arg(vargs, const char *);
    if (max <= len + strnlen(s, max_len)) {
      max = 3 * (len + strnlen(s, max_len)) / 2 + 1;
      out = trealloc(out, char, max);
    }
    strncpy(out + len, s, strnlen(s, max_len));
    len += strnlen(s, max_len);
  }
  out[len] = '\0';

  va_end(vargs);

  return out;
}

int strnlower(char **out, const char *in, size_t max) {
  unsigned len = strnlen(in, max);
  char *wrk = *out = trealloc(*out, char, len + 1);
  for (unsigned i = 0; i < len; i++)
    wrk[i] = tolower(in[i]);
  wrk[len] = '\0';

  return 0;
}

int strntoui(const char *str, size_t size) {
  if (str == NULL)
    return -1;

  char *copy = strndup(str, size), *end_ptr;
  int num = (int)strtol(copy, &end_ptr, 10);
  if (copy == end_ptr || *end_ptr != '\0' || num < 0)
    num = -1;
  tfree(copy);

  return num;
}

size_t pathlen(const char *path) {
  return (size_t)pathconf(path, _PC_PATH_MAX);
}

int MAX(unsigned args, ...) {
  va_list valist;
  va_start(valist, args);
  int max = INT_MIN, cur;
  for (unsigned i = 0; i < args; i++) {
    cur = va_arg(valist, int);
    if (max < cur)
      max = cur;
  }
  va_end(valist);
  return max;
}

int check_null_input_(void *p, const char *func, unsigned line,
                      const char *file) {
  if (!p) {
    return set_log(NOMP_RUNTIME_NULL_INPUT_ENCOUNTERED, NOMP_ERROR,
                   "Input pointer passed to function \"%s\" at line %d in file "
                   "%s is NULL.",
                   func, line, file);
  }
  return 0;
}
