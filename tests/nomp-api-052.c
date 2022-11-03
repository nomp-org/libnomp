#include "nomp-test.h"
#include "nomp.h"

int main(int argc, char *argv[]) {
  char *backend = argc > 1 ? argv[1] : "opencl";
  int device = argc > 2 ? atoi(argv[2]) : 0;
  int platform = argc > 3 ? atoi(argv[3]) : 0;

  const char *valid_knl =
      "void foo(int *a, int N) {                                             \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i;                                                        \n"
      "}                                                                    \n";

  static int id = -1;
  const char *annotations[1] = {0},
             *clauses0[3] = {"transform", "invalid-file:invalid_func", 0},
             *clauses1[3] = {"transform", "nomp-api-50:invalid_transform", 0},
             *clauses2[3] = {"invalid-clause", "nomp-api-50:transform", 0};
  int err = nomp_init(backend, platform, device);

  // Calling nomp_jit with invalid functions should return an error.
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

  // Calling nomp_jit with invalid clauses shoud return an error.
  err = nomp_jit(&id, valid_knl, annotations, clauses2);
  nomp_assert(nomp_get_log_no(err) == NOMP_INVALID_CLAUSE);

  err = nomp_get_log(&desc, err);
  matched =
      match_log(desc, "\\[Error\\] "
                      ".*libnomp\\/src\\/nomp.c:[0-9]* "
                      "Invalid clause is passed into nomp_jit: invalid-clause");
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

  tfree(desc);

  return 0;
}
