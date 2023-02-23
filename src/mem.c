#include "nomp-mem.h"

int nomp_free_(void *p, const char *file, unsigned line) {
  if (p)
    free(p);
  return 0;
}
void *nomp_malloc_(size_t size, const char *file, unsigned line) {
  void *restrict res = malloc(size);
  return res;
}

void *nomp_calloc_(size_t nmemb, size_t size, const char *file, unsigned line) {
  void *restrict res = calloc(nmemb, size);
  return res;
}

void *nomp_realloc_(void *ptr, size_t size, const char *file, unsigned line) {
  void *restrict res = realloc(ptr, size);
  return res;
}
