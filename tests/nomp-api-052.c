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
             *clauses0[4] = {"transform", "invalid-file", "invalid", 0},
             *clauses1[4] = {"transform", "nomp-api-50", "invalid_func", 0},
             *clauses2[4] = {"invalid-clause", "nomp-api-50", "transform", 0},
             *clauses3[4] = {"transform", NULL, "transform", 0},
             *clauses4[4] = {"transform", "nomp-api-50", NULL, 0};
  int err = nomp_init(backend, platform, device);

  // Calling nomp_jit with invalid functions should return an error.
  err = nomp_jit(&id, valid_knl, annotations, clauses0);
  nomp_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILED);

  char *desc;
  err = nomp_get_log_str(&desc, err);
  int matched =
      match_log(desc, "\\[Error\\] "
                      ".*src\\/loopy.c:[0-9]* PyImport_Import() failed when "
                      "importing user transform file: invalid-file.");
  nomp_assert(matched);
  tfree(desc);

  // Invalid transform function
  err = nomp_jit(&id, valid_knl, annotations, clauses1);
  nomp_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILED);

  err = nomp_get_log_str(&desc, err);
  matched = match_log(
      desc, "\\[Error\\] "
            ".*src\\/loopy.c:[0-9]* PyObject_CallFunctionObjArgs() failed when "
            "calling user transform function: invalid_func.");
  nomp_assert(matched);
  tfree(desc);

  // Calling nomp_jit with invalid clauses shoud return an error.
  err = nomp_jit(&id, valid_knl, annotations, clauses2);
  nomp_assert(nomp_get_log_no(err) == NOMP_INVALID_CLAUSE);

  err = nomp_get_log_str(&desc, err);
  matched =
      match_log(desc, "\\[Error\\] "
                      ".*libnomp\\/src\\/nomp.c:[0-9]* "
                      "Invalid clause is passed into nomp_jit: invalid-clause");
  nomp_assert(matched);
  tfree(desc);

  // Missing a semi-colon thus the kernel have a syntax error
  const char *invalid_knl =
      "void foo(int *a, int N) {                                             \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i                                                         \n"
      "}                                                                    \n";

  err = nomp_jit(&id, invalid_knl, annotations, clauses0);
  nomp_assert(nomp_get_log_no(err) == NOMP_LOOPY_CONVERSION_ERROR);

  err = nomp_get_log_str(&desc, err);
  matched = match_log(desc, "\\[Error\\] "
                            ".*"
                            "libnomp\\/src\\/loopy.c:[0-9]* C "
                            "to Loopy conversion failed.");
  nomp_assert(matched);
  tfree(desc);

  // Missing file name should return an error.
  err = nomp_jit(&id, valid_knl, annotations, clauses3);
  nomp_assert(nomp_get_log_no(err) == NOMP_FILE_NAME_NOT_PROVIDED);

  err = nomp_get_log_str(&desc, err);
  matched = match_log(desc, "\\[Error\\] "
                            ".*libnomp\\/src\\/nomp.c:[0-9]* "
                            "File name is not provided.");
  nomp_assert(matched);
  tfree(desc);

  // Missing user callback should return an error.
  err = nomp_jit(&id, valid_knl, annotations, clauses4);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_CALLBACK_NOT_PROVIDED);

  err = nomp_get_log_str(&desc, err);
  matched = match_log(desc, "\\[Error\\] "
                            ".*libnomp\\/src\\/nomp.c:[0-9]* "
                            "User callback function is not provided.");
  nomp_assert(matched);
  tfree(desc);

  err = nomp_finalize();
  nomp_chk(err);

  return 0;
}
