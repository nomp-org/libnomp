#include "nomp.h"
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int log_match(const char *log, const char *pattern) {
  regex_t regex;
  int result = regcomp(&regex, pattern, 0);
  if (!result) {
    result = regexec(&regex, log, 0, NULL, 0);
  }
  regfree(&regex);
  return !result;
}

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device_id = argc > 2 ? atoi(argv[2]) : 0;
  int platform_id = argc > 3 ? atoi(argv[3]) : 0;

  // Calling `nomp_finalize` before `nomp_init` should retrun an error
  int err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_NOT_INITIALIZED_ERROR);

  char *desc;
  err = nomp_get_log(&desc, err);
  int matched = log_match(
      desc,
      "[Error][:space:].*/libnomp/src/nomp.c:[0-9]* Nomp is not initialized.");
  nomp_assert(matched);

  // Calling `nomp_init` twice must return an error, but must not segfault
  err = nomp_init(backend, device_id, platform_id);
  nomp_chk(err);
  err = nomp_init(backend, device_id, platform_id);
  nomp_assert(nomp_get_log_no(err) == NOMP_INITIALIZED_ERROR);

  err = nomp_get_log(&desc, err);
  matched = log_match(
      desc,
      "[Error][:space:].*/libnomp/src/nomp.c:[0-9]* libnomp is already "
      "initialized to "
      "use opencl. Call nomp_finalize() before calling nomp_init() again.");
  nomp_assert(matched);

  // Calling `nomp_finalize` twice must return an error, but must not segfault
  err = nomp_finalize();
  nomp_chk(err);
  err = nomp_finalize();
  nomp_assert(nomp_get_log_no(err) == NOMP_NOT_INITIALIZED_ERROR);

  return 0;
}
