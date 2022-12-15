#include "nomp-impl.h"

char *strcatn(unsigned nstr, unsigned max_len, ...) {
  unsigned n = 0, max = NOMP_BUFSIZ;
  char *out = tcalloc(char, max);

  va_list vargs;
  va_start(vargs, max_len);
  for (int i = 0; i < nstr; i++) {
    const char *s = va_arg(vargs, const char *);
    if (max <= n + strnlen(s, max_len)) {
      max = 3 * (n + strnlen(s, max_len)) / 2 + 1;
      out = trealloc(out, char, max);
    }
    strncpy(out + n, s, strnlen(s, max_len));
    n += strnlen(s, max_len);
  }
  va_end(vargs);
  out[n] = '\0';

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

  char *str_dup = strndup(str, size), *end_ptr;
  int num = (int)strtol(str_dup, &end_ptr, 10);
  if (str_dup == end_ptr || '\0' != *end_ptr || num < 0)
    num = -1;
  tfree(str_dup);

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
