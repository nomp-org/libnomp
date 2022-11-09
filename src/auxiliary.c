#include "nomp-impl.h"

char *strcatn(int nstr, ...) {
  unsigned n = 0, max = NOMP_BUFSIZ;
  char *out = tcalloc(char, max);

  va_list vargs;
  va_start(vargs, nstr);
  for (int i = 0; i < nstr; i++) {
    const char *s = va_arg(vargs, const char *);
    if (max <= n + strlen(s)) {
      max = 3 * (n + strlen(s)) / 2 + 1;
      out = trealloc(out, char, max);
    }
    strncpy(out + n, s, strlen(s));
    n += strlen(s);
  }
  va_end(vargs);
  out[n] = '\0';

  return out;
}

int strnlower(char **out, const char *in, size_t max) {
  unsigned len = strnlen(in, max);
  if (len == max)
    return set_log(NOMP_STR_EXCEED_MAX_LEN, NOMP_ERROR,
                   ERR_STR_EXCEED_MAX_LEN_STR, max);

  char *wrk = *out = trealloc(*out, char, len + 1);
  for (unsigned i = 0; i < len; i++)
    wrk[i] = tolower(in[i]);
  wrk[len] = '\0';

  return 0;
}

int strntoi(const char *str, size_t size) {
  if (str == NULL)
    return -1;
  char *str_dup = strndup(str, size), *end_ptr;
  int num = (int)strtol(str_dup, &end_ptr, 10);
  if (str_dup == end_ptr || '\0' != *end_ptr) {
    tfree(str_dup);
    return -1;
  }
  tfree(str_dup);
  return num >= 0 ? num : -1;
}

size_t pathlen(const char *path) {
  return (size_t)pathconf(path, _PC_PATH_MAX);
}
