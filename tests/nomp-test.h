#if !defined(_NOMP_TEST_H_)
#define _NOMP_TEST_H_

#include "nomp-mem.h"
#include "nomp.h"
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

#define nomp_test_chk(err_id)                                                  \
  {                                                                            \
    if (nomp_get_log_type((err_id)) == NOMP_ERROR)                             \
      return 1;                                                                \
  }

static char *create_knl(const char *knl_fmt, const int args_n, ...) {
  size_t len = strlen(knl_fmt) + args_n * strlen(TOSTRING(test_type)) + 1;
  char *knl = tcalloc(char, len);

  va_list vargs;
  va_start(vargs, args_n);
  vsnprintf(knl, len, knl_fmt, vargs);
  va_end(vargs);

  return knl;
}

static int run_kernel(char *knl, const char **clauses, const int args_n, ...) {
  static int id = -1;

  int err = nomp_jit(&id, knl, clauses);
  nomp_test_chk(err);

  va_list vargs;
  va_start(vargs, args_n);
  err = nomp_run(id, args_n, vargs);
  nomp_test_chk(err);
  va_end(vargs);

  tfree(knl);
  return 0;
}

static int match_log(const char *log, const char *pattern) {
  regex_t regex;
  int result = regcomp(&regex, pattern, 0);
  if (!result)
    result = regexec(&regex, log, 0, NULL, 0);
  regfree(&regex);
  return !result;
}

#endif // _NOMP_TEST_H_
