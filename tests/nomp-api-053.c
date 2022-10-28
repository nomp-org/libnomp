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
  int n = 10;
  int a[10], b[10];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl = "void foo(int *a, int *b, int N) {                      \n"
                    "  for (int i = 0; i < N; i++)                          \n"
                    "    a[i] = a[i] * b[i];                                \n"
                    "}                                                      \n";

  int err = nomp_init(backend, device, platform);
  err = nomp_update(a, 0, n, sizeof(int), NOMP_TO);
  nomp_chk(err);

  static int id = -1;
  const char *annotations[1] = {0},
             *clauses[3] = {"transform", "nomp-api-50:transform", 0};
  err = nomp_jit(&id, knl, annotations, clauses);
  nomp_chk(err);

  // Invoke with invalid kernel_id
  err = nomp_run(-1, 3, "a", NOMP_PTR, sizeof(int), a, "b", NOMP_PTR,
                 sizeof(int), b, "N", NOMP_INTEGER, sizeof(int), &n);
  nomp_assert(nomp_get_log_no(err) == NOMP_INVALID_KNL);
  char *desc;
  err = nomp_get_log(&desc, err);
  int matched = match_log(desc, "\\[Error\\] "
                                ".*libnomp\\/"
                                "src\\/nomp.c:[0-9]* Invalid kernel -1.");
  nomp_assert(matched);

  // Invoke fails because b is not mapped
  err = nomp_run(id, 3, "a", NOMP_PTR, sizeof(int), a, "b", NOMP_PTR,
                 sizeof(int), b, "N", NOMP_INTEGER, sizeof(int), &n);
  nomp_assert(nomp_get_log_no(err) == NOMP_KNL_RUN_ERROR);
  err = nomp_get_log(&desc, err);
  matched = match_log(desc, "\\[Error\\] "
                            ".*\\/libnomp\\/"
                            "src\\/nomp.c:[0-9]* Kernel 0 run failed.");
  nomp_assert(matched);

  err = nomp_finalize();
  nomp_chk(err);

  free(desc);

  return 0;
}
