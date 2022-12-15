#include "nomp-test.h"
#include "nomp.h"

const char *valid_knl =
    "void foo(int *a, int N) {                                            \n"
    "  for (int i = 0; i < N; i++)                                        \n"
    "    a[i] = i;                                                        \n"
    "}                                                                    \n";

// Calling nomp_jit with invalid clauses should return an error.
static int test_invalid_clause() {
  const char *clauses[4] = {"invalid-clause", "transforms", "foo", 0};
  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", NOMP_PTR, sizeof(int),
                     "N", NOMP_INT, sizeof(int));
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_IS_INVALID);

  char *desc = nomp_get_log_str(err);
  int matched = match_log(
      desc,
      "\\[Error\\] "
      ".*libnomp\\/src\\/nomp.c:[0-9]* "
      "Clause \"invalid-clause\" passed into nomp_jit is not a valid caluse.");
  nomp_test_assert(matched);
  tfree(desc);

  return 0;
}

// Calling nomp_jit with invalid function should return an error.
static int test_call_jit_with_invalid_function() {
  const char *clauses[4] = {"transform", "transforms", "invalid_func", 0};
  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", NOMP_PTR, sizeof(int),
                     "N", NOMP_INT, sizeof(int));
  nomp_test_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILED);

  char *desc = nomp_get_log_str(err);
  int matched =
      match_log(desc, "\\[Error\\] .*src\\/loopy.c:[0-9]* Calling "
                      "user transform function: \"invalid_func\" failed.");
  nomp_test_assert(matched);
  tfree(desc);

  return 0;
}

// Calling nomp_jit with invalid transform script should return an error.
static int test_call_jit_with_invalid_script() {
  const char *clauses[4] = {"transform", "invalid_script", "foo", 0};
  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", NOMP_PTR, sizeof(int),
                     "N", NOMP_INT, sizeof(int));
  nomp_test_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILED);

  char *desc = nomp_get_log_str(err);
  int matched = match_log(desc, "\\[Error\\] .*src\\/loopy.c:[0-9]* Calling "
                                "user transform function: \"foo\" failed.");
  nomp_test_assert(matched);
  tfree(desc);

  return 0;
}
// Calling nomp_jit with missing file name should return an error.
static int test_missing_filename() {
  const char *clauses[4] = {"transform", NULL, "transform", 0};
  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", NOMP_PTR, sizeof(int),
                     "N", NOMP_INT, sizeof(int));
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_NOT_PROVIDED);

  char *desc = nomp_get_log_str(err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* "
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");
  nomp_test_assert(matched);
  tfree(desc);

  return 0;
}

// Calling nomp_jit with missing user callback should return an error.
static int test_missing_user_callback() {
  const char *clauses[4] = {"transform", "transforms", NULL, 0};
  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", NOMP_PTR, sizeof(int),
                     "N", NOMP_INT, sizeof(int));
  nomp_test_assert(nomp_get_log_no(err) == NOMP_USER_INPUT_NOT_PROVIDED);

  char *desc = nomp_get_log_str(err);
  int matched = match_log(
      desc, "\\[Error\\] "
            ".*libnomp\\/src\\/nomp.c:[0-9]* "
            "\"transform\" clause should be followed by a file name and a "
            "function name. At least one of them is not provided.");
  nomp_test_assert(matched);
  tfree(desc);

  return 0;
}

// Calling nomp_jit with a kernel having a syntax error should be an error.
static int test_syntax_error_kernel() {
  const char *invalid_knl =
      "void foo(int *a, int N) {                                            \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i                                                         \n"
      "}                                                                    \n";
  const char *clauses[4] = {"transform", "transforms", "foo", 0};
  static int id = -1;
  int err = nomp_jit(&id, invalid_knl, clauses, 2, "a", NOMP_PTR, sizeof(int),
                     "N", NOMP_INT, sizeof(int));
  nomp_test_assert(nomp_get_log_no(err) == NOMP_LOOPY_CONVERSION_ERROR);

  char *desc = nomp_get_log_str(err);
  int matched = match_log(desc, "\\[Error\\] "
                                ".*"
                                "libnomp\\/src\\/loopy.c:[0-9]* C "
                                "to Loopy conversion failed.");
  nomp_test_assert(matched);
  tfree(desc);

  return 0;
}

// Calling nomp_jit with valid parameters should succeed.
static int test_valid_params() {
  const char *clauses[4] = {"transform", "transforms", "foo", 0};
  static int id = -1;
  int err = nomp_jit(&id, valid_knl, clauses, 2, "a", NOMP_PTR, sizeof(int),
                     "N", NOMP_INT, sizeof(int));
  nomp_test_chk(err);

  return 0;
}

int main(int argc, const char *argv[]) {
  int err = nomp_init(argc, argv);
  nomp_test_chk(err);

  err |= SUBTEST(test_invalid_clause);
  err |= SUBTEST(test_call_jit_with_invalid_function);
  err |= SUBTEST(test_call_jit_with_invalid_script);
  err |= SUBTEST(test_missing_filename);
  err |= SUBTEST(test_missing_user_callback);
  err |= SUBTEST(test_syntax_error_kernel);
  err |= SUBTEST(test_valid_params);

  err = nomp_finalize();
  nomp_test_chk(err);

  return err;
}
