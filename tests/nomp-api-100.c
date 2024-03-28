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

#define TEST_IMPL_H "nomp-api-100-impl.h"
#include "nomp-generate-tests.h"
#undef TEST_IMPL_H

// Calling nomp_jit() with invalid file should return an error.
static int test_invalid_transform_file(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(100_invalid_file, );
  return err;
}

// Calling nomp_jit() with invalid function should return an error.
static int test_invalid_transform_function(void) {
  int err = 0;
  TEST_BUILTIN_TYPES(100_invalid_function, );
  return err;
}

// Calling nomp_jit() with empty file name should return an error.
static int test_empty_filename(void) {
  const char *clauses[4] = {"transform", NULL, "tile", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_err_str(err);
  int   eq  = logcmp(log, "\\[Error\\] .*src\\/.*.c:[0-9]* Module "
                             "name and/or function name not provided.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Calling nomp_jit() with empty transform function name should return an error.
static int test_empty_user_callback(void) {
  const char *clauses[4] = {"transform", "nomp_api_100", NULL, 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_err_str(err);
  int   eq  = logcmp(log, "\\[Error\\] .*src\\/.*.c:[0-9]* Module "
                             "name and/or function name not provided.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Calling nomp_jit() with invalid clauses should return an error.
static int test_invalid_clause(void) {
  const char *clauses[4] = {"invalid-clause", "nomp_api_100", "tile", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *log = nomp_get_err_str(err);
  int   eq  = logcmp(
      log, "\\[Error\\] .*src\\/.*.c:[0-9]* Clause "
              "\"invalid-clause\" passed into nomp_jit is not a valid clause.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Calling nomp_jit() with a kernel which has a syntax error should fail.
static int test_syntax_error_in_kernel(void) {
  static int  id          = -1;
  const char *clauses0[4] = {"transform", "nomp_api_100", "tile", 0};
  int err = nomp_jit(&id, invalid_knl, clauses0, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_LOOPY_CONVERSION_FAILURE);

  char *log = nomp_get_err_str(err);
  int eq = logcmp(log, "\\[Error\\] .*src\\/.*.c:[0-9]* Converting C source to "
                       "loopy kernel failed.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

// Calling nomp_jit() with a transform function with a syntax error should
// return an error.
static int test_syntax_error_in_transform_function(void) {
  const char *clauses[4] = {"transform", "nomp_api_100",
                            "function_with_syntax_error", 0};

  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", sizeof(int), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_PY_CALL_FAILURE);

  char *log = nomp_get_err_str(err);
  int   eq  = logcmp(
      log,
      "\\[Error\\] .*src\\/.*.c:[0-9]* Calling Python function "
         "\"function_with_syntax_error\" from module \"nomp_api_100\" failed.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}

int main(int argc, const char *argv[]) {
  nomp_test_check(nomp_init(argc, argv));

  int err = 0;
  err |= SUBTEST(test_invalid_transform_file);
  err |= SUBTEST(test_invalid_transform_function);
  err |= SUBTEST(test_invalid_clause);
  err |= SUBTEST(test_empty_filename);
  err |= SUBTEST(test_empty_user_callback);
  err |= SUBTEST(test_syntax_error_in_kernel);
  err |= SUBTEST(test_syntax_error_in_transform_function);

  nomp_test_check(nomp_finalize());

  return err;
}
