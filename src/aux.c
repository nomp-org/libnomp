#include "nomp-impl.h"

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

int nomp_check_py_script_path(const char *path) {
  char *py = nomp_str_cat(2, PATH_MAX, path, ".py");
  int err = nomp_path_len(NULL, (const char *)py);
  nomp_free(&py);
  return err;
}

char *nomp_copy_env(const char *name, size_t size) {
  const char *tmp = getenv(name);
  if (tmp)
    return strndup(tmp, size);
  return NULL;
}
