#include "nomp-test.h"

static const char *valid_knl =
    "void foo(int *a, int N) {                                            \n"
    "  for (int i = 0; i < N; i++)                                        \n"
    "    a[i] = i;                                                        \n"
    "}                                                                    \n";

// Calling nomp_jit() with invalid file should return an error.
static int test_invalid_file() {
  const char *clauses[4] = {"transform", "invalid-file", "invalid", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched =
      match_log(desc, "\\[Error\\] "
                      ".*src\\/loopy.c:[0-9]* PyImport_Import() failed when "
                      "importing user transform file: invalid-file.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

// Calling nomp_jit() with invalid function should return an error.
static int test_invalid_transform_function() {
  const char *clauses[4] = {"transform", "nomp-api-100", "invalid_func", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*src\\/loopy.c:[0-9]* PyObject_CallFunctionObjArgs() failed when "
            "calling user transform function: invalid_func.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

// Calling nomp_jit() with invalid clauses should return an error.
static int test_invalid_clause() {
  const char *clauses[4] = {"invalid-clause", "nomp-api-100", "transform", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc,
      "\\[Error\\] "
      ".*libnomp\\/src\\/nomp.c:[0-9]* "
      "Clause \"invalid-clause\" passed into nomp_jit is not a valid caluse.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

// Missing file name in nomp_jit() should return an error.
static int test_missing_filename() {
  const char *clauses[4] = {"transform", NULL, "transform", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_NOT_PROVIDED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* "
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

// Missing user callback should return an error.
static int test_missing_user_callback() {
  const char *clauses[4] = {"transform", "nomp-api-100", NULL, 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_NOT_PROVIDED);

  char *desc;
  nomp_get_log_str(&desc, err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* "
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

// The kernel has a syntax error due to a missing a semicolon.
static int test_syntax_error_in_kernel() {
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
  nomp_get_log_str(&desc, err);
  int matched = match_log(desc, "\\[Error\\] "
                                ".*"
                                "libnomp\\/src\\/loopy.c:[0-9]* C "
                                "to Loopy conversion failed.");
  tfree(desc);
  nomp_test_assert(matched);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err |= SUBTEST(test_invalid_file);
  err |= SUBTEST(test_invalid_transform_function);
  err |= SUBTEST(test_invalid_clause);
  err |= SUBTEST(test_missing_filename);
  err |= SUBTEST(test_missing_user_callback);
  err |= SUBTEST(test_syntax_error_in_kernel);

  err = nomp_finalize();
  nomp_test_chk(err);

  return err;
}
