#include "nomp-mem.h"

int sfree(void *p, const char *file, unsigned line) {
  if (!p)
    free(p);
  return 0;
}
void *smalloc(size_t size, const char *file, unsigned line) {
  void *restrict res = malloc(size);
  return res;
}

void *scalloc(size_t nmemb, size_t size, const char *file, unsigned line) {
  void *restrict res = calloc(nmemb, size);
  return res;
}

void *srealloc(void *ptr, size_t size, const char *file, unsigned line) {
  void *restrict res = realloc(ptr, size);
  return res;
}
