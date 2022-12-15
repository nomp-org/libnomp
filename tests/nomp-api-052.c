#include "nomp-test.h"

const char *valid_knl =
    "void foo(int *a, int N) {                                            \n"
    "  for (int i = 0; i < N; i++)                                        \n"
    "    a[i] = i;                                                        \n"
    "}                                                                    \n";

// Calling nomp_jit with invalid functions should return an error.
static int test_call_jit_with_invalid_function(int argc, const char **argv) {

  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  static int id = -1;
  const char *clauses0[4] = {"transform", "invalid-file", "invalid", 0};
  err = nomp_jit(&id, valid_knl, clauses0);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched =
      match_log(desc, "\\[Error\\] "
                      ".*src\\/loopy.c:[0-9]* PyImport_Import() failed when "
                      "importing user transform file: invalid-file.");
  nomp_test_assert(matched);
  tfree(desc);
  return 0;
}

// Invalid transform function
static int test_invalid_transform_function() {
  const char *clauses1[4] = {"transform", "nomp-api-50", "invalid_func", 0};
  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses1);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*src\\/loopy.c:[0-9]* PyObject_CallFunctionObjArgs() failed when "
            "calling user transform function: invalid_func.");
  nomp_test_assert(matched);
  tfree(desc);
  return 0;
}

// Calling nomp_jit with invalid clauses should return an error.
static int test_invalid_clause() {
  const char *clauses2[4] = {"invalid-clause", "nomp-api-50", "transform", 0};
  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses2);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc,
      "\\[Error\\] "
      ".*libnomp\\/src\\/nomp.c:[0-9]* "
      "Clause \"invalid-clause\" passed into nomp_jit is not a valid caluse.");
  nomp_test_assert(matched);
  tfree(desc);
  return 0;
}

// Missing file name should return an error.
static int test_missing_filename() {
  const char *clauses3[4] = {"transform", NULL, "transform", 0};
  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses3);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_NOT_PROVIDED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* "
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");

  nomp_test_assert(matched);
  tfree(desc);
  return 0;
}

// Missing user callback should return an error.
static int tset_missing_user_callback() {
  const char *clauses4[4] = {"transform", "nomp-api-50", NULL, 0};
  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses4);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_NOT_PROVIDED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* "
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");
  nomp_test_assert(matched);
  tfree(desc);
  return 0;
}

// The kernel has a syntax error due to a missing a semicolon.
static int test_syntax_error_kernel() {
  const char *invalid_knl =
      "void foo(int *a, int N) {                                            \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i                                                         \n"
      "}                                                                    \n";
  static int id = -1;
  const char *clauses0[4] = {"transform", "invalid-file", "invalid", 0};
  int err = nomp_jit(&id, invalid_knl, clauses0);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_LOOPY_CONVERSION_ERROR);

  char *desc;
  err = nomp_get_log_str(&desc, err);
  int matched = match_log(desc, "\\[Error\\] "
                                ".*"
                                "libnomp\\/src\\/loopy.c:[0-9]* C "
                                "to Loopy conversion failed.");
  nomp_test_assert(matched);
  tfree(desc);

  err = nomp_finalize();
  nomp_test_chk(err);
  return 0;
}

int main(int argc, const char *argv[]) {
  int err = 0;

  err |= SUBTEST(test_call_jit_with_invalid_function, argc, argv);
  err |= SUBTEST(test_invalid_transform_function);
  err |= SUBTEST(test_invalid_clause);
  err |= SUBTEST(test_missing_filename);
  err |= SUBTEST(tset_missing_user_callback);
  err |= SUBTEST(test_syntax_error_kernel);

  return err;
}
