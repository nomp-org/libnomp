#if !defined(_NOMP_TEST_H_)
#define _NOMP_TEST_H_

#include "nomp-mem.h"
#include "nomp.h"
#include <math.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOKEN_PASTE_(a, b) a##b
#define TOKEN_PASTE(a, b) TOKEN_PASTE_(a, b)

#define TOSTRING_(x) #x
#define TOSTRING(x) TOSTRING_(x)

#define TEST_BUILTIN_TYPES(a, ...)                                             \
  {                                                                            \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _int)(__VA_ARGS__);                 \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _long)(__VA_ARGS__);                \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _unsigned)(__VA_ARGS__);            \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _unsigned_long)(__VA_ARGS__);       \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _double)(__VA_ARGS__);              \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _float)(__VA_ARGS__);               \
  }

#define SUBTEST(subtest, ...) subtest_(subtest(__VA_ARGS__), TOSTRING(subtest))
static int subtest_(int err, const char *test_name) {
  char *result = err ? "\033[31mFailed" : "\033[32mPassed";
  printf("\t%s: %s\033[0m\n", test_name, result);
  return err;
}

static void nomp_test_assert(int *output, int cond) {
  if (!cond && *output == 0) {
    *output = 1;
  }
}

static void nomp_test_chk(int *output, int err_id) {
  if (nomp_get_log_type(err_id) == NOMP_ERROR && *output == 0) {
    *output = 1;
  }
}

static int match_log(const char *log, const char *pattern) {
  regex_t regex;
  int result = regcomp(&regex, pattern, 0);
  if (!result)
    result = regexec(&regex, log, 0, NULL, 0);
  regfree(&regex);
  return !result;
}

static int parse_input(int argc, char *argv[], char **backend, int *device,
                       int *platform) {
  *backend = argc > 1 ? argv[1] : "opencl";
  *device = argc > 2 ? atoi(argv[2]) : 0;
  *platform = argc > 3 ? atoi(argv[3]) : 0;

  return 0;
}

#endif // _NOMP_TEST_H_
