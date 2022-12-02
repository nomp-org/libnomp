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
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _int(__VA_ARGS__);)                 \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _long(__VA_ARGS__);)                \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _unsigned(__VA_ARGS__);)            \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _unsigned_long(__VA_ARGS__);)       \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _double(__VA_ARGS__);)              \
    TOKEN_PASTE(TOKEN_PASTE(nomp_api_, a), _float(__VA_ARGS__);)               \
  }

#define PRINT_SUBTEST(output, subtest, ...)                                    \
  {                                                                            \
    int err = (subtest)(__VA_ARGS__);                                          \
    char *result = err ? "\033[31mFailed" : "\033[32mPassed";                  \
    printf("\t%s: %s\033[0m\n", TOSTRING(subtest), result);                    \
    print_error(err);                                                          \
    output |= err;                                                             \
  }

#define MAX_NOMP_ERROR_DETAILS 200
static char *error_detail;

#define nomp_test_assert(output, cond)                                         \
  nomp_test_assert_(output, cond, __FILE__, __LINE__)
static void nomp_test_assert_(int *output, int cond, const char *file,
                              unsigned line) {
  if (!cond && *output == 0) {
    error_detail = tcalloc(char, MAX_NOMP_ERROR_DETAILS);
    snprintf(error_detail, MAX_NOMP_ERROR_DETAILS,
             "\tnomp_assert failure at %s:%d\n\n", file, line);
    *output = 1;
  }
}

#define nomp_test_chk(output, err)                                             \
  nomp_test_chk_(output, err, __FILE__, __LINE__)
static void nomp_test_chk_(int *output, int err_id, const char *file,
                           unsigned line) {
  if (nomp_get_log_type(err_id) == NOMP_ERROR && *output == 0) {
    char *err_str;
    nomp_get_log_str(&err_str, err_id);
    printf("\t%s:%d %s\n\n", file, line, err_str);
    tfree(err_str);
    *output = 1;
  }
}

static void print_error(int output) {
  if (output == 1) {
    printf("%s", error_detail);
    tfree(error_detail);
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
