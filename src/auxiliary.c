#include "nomp-impl.h"

char *strcatn(int nstr, ...) {
  unsigned n = 0, max = BUFSIZ;
  char *out = (char *)calloc(max, sizeof(char));

  va_list vargs;
  va_start(vargs, nstr);
  for (int i = 0; i < nstr; i++) {
    const char *s = va_arg(vargs, const char *);
    if (max <= n + strlen(s)) {
      max = 3 * (n + strlen(s)) / 2 + 1;
      out = realloc(out, sizeof(char) * max);
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
    return 1;

  char *wrk = *out = (char *)realloc(*out, (len + 1) * sizeof(char));
  for (unsigned i = 0; i < len; i++)
    wrk[i] = tolower(in[i]);
  wrk[len] = '\0';

  return 0;
}
