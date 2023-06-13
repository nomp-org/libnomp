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

// Calling nomp_jit() with invalid file should return an error.
static int test_invalid_file() {
  const char *clauses[4] = {"transform", "invalid-file", "invalid", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_log_str(err);
  int eq = logcmp(log, "\\[Error\\] .*src\\/aux.c:[0-9]* Unable to find path: "
                       "\"invalid-file.py\".");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Calling nomp_jit() with invalid function should return an error.
static int test_invalid_transform_function() {
  const char *clauses[4] = {"transform", "nomp-api-100", "invalid_func", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILURE);

  char *log = nomp_get_log_str(err);
  int eq = logcmp(
      log, "\\[Error\\] .*src\\/loopy.c:[0-9]* Failed to call user transform "
           "function: \"invalid_func\" in file: \"nomp-api-100\".");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Calling nomp_jit() with invalid clauses should return an error.
static int test_invalid_clause() {
  const char *clauses[4] = {"invalid-clause", "nomp-api-100", "transform", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_log_str(err);
  int eq = logcmp(
      log, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* Clause "
           "\"invalid-clause\" passed into nomp_jit is not a valid clause.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Missing file name in nomp_jit() should return an error.
static int test_missing_filename() {
  const char *clauses[4] = {"transform", NULL, "transform", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_log_str(err);
  int eq =
      logcmp(log, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* \"transform\" "
                  "clause should be followed by a file name and a function "
                  "name. At least one of them is not provided.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Missing user callback should return an error.
static int test_missing_user_callback() {
  const char *clauses[4] = {"transform", "nomp-api-100", NULL, 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_log_str(err);
  int eq =
      logcmp(log, "\\[Error\\] .*libnomp\\/src\\/nomp.c:[0-9]* \"transform\" "
                  "clause should be followed by a file name and a function "
                  "name. At least one of them is not provided.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// The kernel has a syntax error due to a missing a semicolon.
static int test_syntax_error_in_kernel() {
  static int id = -1;
  const char *clauses0[4] = {"transform", "nomp-api-100", "transform", 0};
  int err = nomp_jit(&id, invalid_knl, clauses0, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_log_no(err) == NOMP_LOOPY_CONVERSION_FAILURE);

  char *log = nomp_get_log_str(err);
  int eq =
      logcmp(log, "\\[Error\\] .*libnomp\\/src\\/loopy.c:[0-9]* C to Loopy "
                  "conversion failed.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_check(err);

  err |= SUBTEST(test_invalid_file);
  err |= SUBTEST(test_invalid_transform_function);
  err |= SUBTEST(test_invalid_clause);
  err |= SUBTEST(test_missing_filename);
  err |= SUBTEST(test_missing_user_callback);
  err |= SUBTEST(test_syntax_error_in_kernel);

  err = nomp_finalize();
  nomp_check(err);

  return err;
}
