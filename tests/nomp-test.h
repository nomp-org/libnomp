#if !defined(_NOMP_TEST_H_)
#define _NOMP_TEST_H_

#include "nomp-mem.h"
#include <math.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOKEN_PASTE_(a, b) a##b
#define TOKEN_PASTE(a, b) TOKEN_PASTE_(a, b)

#define TOSTRING_(x) #x
#define TOSTRING(x) TOSTRING_(x)

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
