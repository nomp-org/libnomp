#include "nomp-test.h"

static const char *valid_knl =
    "void foo(int *a, int N) {                                            \n"
    "  for (int i = 0; i < N; i++)                                        \n"
    "    a[i] = i;                                                        \n"
    "}                                                                    \n";

static const char *invalid_knl =
    "void foo(int *a, int N) {                                            \n"
    "  for (int i = 0; i < N; i++)                                        \n"
    "    a[i] = i                                                         \n"
    "}                                                                    \n";

// Calling nomp_jit() with missing file should return an error.
static int test_missing_transform_file(void) {
  const char *clauses[4] = {"transform", "missing-file", "missing", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_err_str(err);
  int eq = logcmp(log, "\\[Error\\] .*src\\/loopy.c:[0-9]* Python module "
                       "\"missing-file\" not found.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Calling nomp_jit() with missing function should return an error.
static int test_missing_transform_function(void) {
  const char *clauses[4] = {"transform", "nomp-api-100", "missing_func", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_err_str(err);
  int eq =
      logcmp(log, "\\[Error\\] .*src\\/loopy.c:[0-9]* Python function "
                  "\"missing_func\" not found in module \"nomp-api-100\".");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Calling nomp_jit() with invalid clauses should return an error.
static int test_invalid_clause(void) {
  const char *clauses[4] = {"invalid-clause", "nomp-api-100", "transform", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_err_str(err);
  int eq = logcmp(
      log, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* Clause "
           "\"invalid-clause\" passed into nomp_jit is not a valid clause.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Empty file name in nomp_jit() should return an error.
static int test_empty_filename(void) {
  const char *clauses[4] = {"transform", NULL, "transform", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_err_str(err);
  int eq =
      logcmp(log, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* \"transform\" "
                  "clause should be followed by a file name and a function "
                  "name. At least one of them is not provided.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Empty user callback in nomp_jit() should return an error.
static int test_empty_user_callback(void) {
  const char *clauses[4] = {"transform", "nomp-api-100", NULL, 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_err_str(err);
  int eq =
      logcmp(log, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* \"transform\" "
                  "clause should be followed by a file name and a function "
                  "name. At least one of them is not provided.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// The kernel has a syntax error due to a missing a semicolon.
static int test_syntax_error_in_kernel(void) {
  static int id = -1;
  const char *clauses0[4] = {"transform", "nomp-api-100", "transform", 0};
  int err = nomp_jit(&id, invalid_knl, clauses0, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_LOOPY_CONVERSION_FAILURE);

  char *log = nomp_get_err_str(err);
  int eq =
      logcmp(log, "\\[Error\\] .*libnomp\\/src\\/loopy.c:[0-9]* C to Loopy "
                  "conversion failed.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Calling nomp_jit() with a transform function with a syntax error should
// return an error.
static int test_syntax_error_in_transform_function(void) {
  const char *clauses[4] = {"transform", "nomp-api-100",
                            "function_with_syntax_error", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_PY_CALL_FAILURE);

  char *log = nomp_get_err_str(err);
  int eq = logcmp(
      log,
      "\\[Error\\] .*src\\/loopy.c:[0-9]* Failed to call user transform "
      "function: \"function_with_syntax_error\" in file: \"nomp-api-100\".");

  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_check(err);

  err |= SUBTEST(test_missing_transform_file);
  err |= SUBTEST(test_missing_transform_function);
  err |= SUBTEST(test_invalid_clause);
  err |= SUBTEST(test_empty_filename);
  err |= SUBTEST(test_empty_user_callback);
  err |= SUBTEST(test_syntax_error_in_kernel);
  err |= SUBTEST(test_syntax_error_in_transform_function);

  err = nomp_finalize();
  nomp_test_check(err);

  return err;
}
