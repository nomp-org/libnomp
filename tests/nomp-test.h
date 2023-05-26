#if !defined(_NOMP_TEST_H_)
#define _NOMP_TEST_H_

#define NOMP_TEST_MAX_BUFSIZ 4096

#include "nomp-aux.h"
#include "nomp-mem.h"
#include "nomp.h"
#include <assert.h>
#include <math.h>
#include <regex.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOKEN_PASTE_(a, b) a##b
#define TOKEN_PASTE(a, b) TOKEN_PASTE_(a, b)

#define TOSTRING_(x) #x
#define TOSTRING(x) TOSTRING_(x)

#if defined(TEST_INT_ONLY)
#define TEST_BUILTIN_TYPES(api, ...)                                           \
  {                                                                            \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _int)(__VA_ARGS__);        \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _long)(__VA_ARGS__);       \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _unsigned)(__VA_ARGS__);   \
    err |=                                                                     \
        TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _unsigned_long)(__VA_ARGS__); \
  }
#else
#define TEST_BUILTIN_TYPES(api, ...)                                           \
  {                                                                            \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _int)(__VA_ARGS__);        \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _long)(__VA_ARGS__);       \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _unsigned)(__VA_ARGS__);   \
    err |=                                                                     \
        TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _unsigned_long)(__VA_ARGS__); \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _double)(__VA_ARGS__);     \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _float)(__VA_ARGS__);      \
  }
#endif

static int subtest_(int err, const char *test_name) {
  char *result = err ? "\033[31mFailed" : "\033[32mPassed";
  printf("\t%s: %s\033[0m\n", test_name, result);
  return err;
}
#define SUBTEST(subtest, ...) subtest_(subtest(__VA_ARGS__), TOSTRING(subtest))

#define nomp_test_assert(cond)                                                 \
  {                                                                            \
    if (!(cond))                                                               \
      return 1;                                                                \
  }

#define nomp_test_chk(err)                                                     \
  {                                                                            \
    if (nomp_get_log_type((err)) == NOMP_ERROR)                                \
      return err;                                                              \
  }

static char *generate_knl(const char *fmt, int nargs, ...) {
  size_t len = strlen(fmt) + 1;

  va_list vargs;
  va_start(vargs, nargs);
  for (unsigned i = 0; i < nargs; i++)
    len += strlen(va_arg(vargs, const char *));
  va_end(vargs);

  char *knl = nomp_calloc(char, len);

  va_start(vargs, nargs);
  vsnprintf(knl, len, fmt, vargs);
  va_end(vargs);

  return knl;
}

static int logcmp(const char *log, const char *pattern) {
  regex_t regex;
  int result = regcomp(&regex, pattern, 0);
  if (!result)
    result = regexec(&regex, log, 0, NULL, 0);
  regfree(&regex);
  return !result;
}

#endif // _NOMP_TEST_H_
