#include "nomp-test.h"

#define nomp_api_102_aux TOKEN_PASTE(nomp_api_102_aux, TEST_SUFFIX)
static int nomp_api_102_aux(const char **clauses) {
  const char *fmt =
      "void foo(%s *a, int N) {                                             \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i;                                                        \n"
      "}                                                                    \n";

  char *knl = generate_knl(fmt, 1, TOSTRING(TEST_TYPE));
  int id = -1;
  int err = nomp_jit(&id, knl, clauses, 2, "a", sizeof(TEST_TYPE), NOMP_PTR,
                     "N", sizeof(int), NOMP_INT);
  nomp_free(&knl);
  return err;
}

#define nomp_api_102_valid TOKEN_PASTE(nomp_api_102_valid, TEST_SUFFIX)
static int nomp_api_102_valid() {
  const char *clauses[4] = {"transform", "nomp-api-102", "transform", 0};
  nomp_check(nomp_api_102_aux(clauses));

  return 0;
}
#undef nomp_api_102_valid

#define nomp_api_102_invalid TOKEN_PASTE(nomp_api_102_invalid, TEST_SUFFIX)
static int nomp_api_102_invalid() {
  const char *clauses[4] = {"transform", "nomp-api-102", "invalid_func", 0};
  int err = nomp_api_102_aux(clauses);

  nomp_test_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILURE);
  char *log = nomp_get_log_str(err);
  int eq = logcmp(
      log, "\\[Error\\] .*src\\/lpy.c:[0-9]* Failed to call user transform "
           "function: \"invalid_func\" in file: \"nomp-api-102\".");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}
#undef nomp_api_102_invalid
#undef nomp_api_102_aux
