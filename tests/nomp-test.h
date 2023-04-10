#if !defined(_NOMP_TEST_H_)
#define _NOMP_TEST_H_

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

#ifndef TEST_INT_ONLY
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
#else
#define TEST_BUILTIN_TYPES(api, ...)                                           \
  {                                                                            \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _int)(__VA_ARGS__);        \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _long)(__VA_ARGS__);       \
    err |= TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _unsigned)(__VA_ARGS__);   \
    err |=                                                                     \
        TOKEN_PASTE(TOKEN_PASTE(nomp_api_, api), _unsigned_long)(__VA_ARGS__); \
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

static int create_knl(int *id, const char *knl_fmt, const char **clauses,
                      const int args_n, ...) {
  size_t len = strlen(knl_fmt) + args_n * strlen(TOSTRING(TEST_TYPE)) + 1;
  char knl[len];

  va_list vargs;
  va_start(vargs, args_n);
  vsnprintf(knl, len, knl_fmt, vargs);
  va_end(vargs);

  nomp_test_chk(nomp_jit(id, knl, clauses));
  return 0;
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
