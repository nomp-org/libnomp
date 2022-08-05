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
