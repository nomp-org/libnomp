#include <stddef.h>
#include <stdlib.h>

int sfree(void *p, const char *file, unsigned line);
#define tfree(x) sfree(x, __FILE__, __LINE__)

void *smalloc(size_t size, const char *file, unsigned line);
#define tmalloc(T, count)                                                      \
  ((T *)smalloc((count) * sizeof(T), __FILE__, __LINE__))

void *scalloc(size_t nmemb, size_t size, const char *file, unsigned line);
#define tcalloc(T, count) ((T *)scalloc((count), sizeof(T), __FILE__, __LINE__))

void *srealloc(void *ptr, size_t size, const char *file, unsigned line);
#define trealloc(ptr, T, count)                                                \
  ((T *)srealloc((ptr), (count) * sizeof(T), __FILE__, __LINE__))
