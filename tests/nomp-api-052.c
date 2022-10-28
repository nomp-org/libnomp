#include "nomp.h"
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int match_log(const char *log, const char *pattern) {
  regex_t regex;
  int result = regcomp(&regex, pattern, 0);
  if (!result)
    result = regexec(&regex, log, 0, NULL, 0);
  regfree(&regex);
  return !result;
}

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  const char *valid_knl =
      "void foo(int *a, int N) {                                             \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i;                                                        \n"
      "}                                                                    \n";

  // Calling nomp_jit with invalid functions should return an error.
  static int id = -1;
  const char *annotations[1] = {0},
             *clauses0[3] = {"transform", "invalid-file:invalid_func", 0},
             *clauses1[3] = {"transform", "nomp-api-50:invalid_transform", 0};
  int err = nomp_init(backend, device, platform);
  err = nomp_jit(&id, valid_knl, annotations, clauses0);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_CALLBACK_NOT_FOUND);

  char *desc;
  err = nomp_get_log(&desc, err);
  int matched =
      match_log(desc, "\\[Error\\] "
                      ".*libnomp\\/"
                      "src\\/loopy.c:[0-9]* Specified "
                      "user callback function not found in file invalid-file.");
  nomp_assert(matched);

  // Invalid transform function
  err = nomp_jit(&id, valid_knl, annotations, clauses1);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_CALLBACK_FAILURE);

  err = nomp_get_log(&desc, err);
  matched = match_log(desc, "\\[Error\\] "
                            ".*libnomp\\/src\\/loopy.c:[0-9]* "
                            "User callback function invalid_transform failed.");
  nomp_assert(matched);

  // Missing a semi-colon thus the kernel have a syntax error
  const char *invalid_knl =
      "void foo(int *a, int N) {                                             \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i                                                         \n"
      "}                                                                    \n";

  err = nomp_jit(&id, invalid_knl, annotations, clauses0);
  nomp_assert(nomp_get_log_no(err) == NOMP_LOOPY_CONVERSION_ERROR);

  err = nomp_get_log(&desc, err);
  matched = match_log(desc, "\\[Error\\] "
                            ".*"
                            "libnomp\\/src\\/loopy.c:[0-9]* C "
                            "to Loopy conversion failed.");
  nomp_assert(matched);

  err = nomp_finalize();
  nomp_chk(err);

  free(desc);

  return 0;
}
