#include "nomp-test.h"

#define nomp_api_100_aux TOKEN_PASTE(nomp_api_100_aux, TEST_SUFFIX)
static int nomp_api_100_aux(const char **clauses) {
  const char *fmt =
      "void foo(%s *a, int N) {                                             \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i;                                                        \n"
      "}                                                                    \n";

  char *knl = generate_knl(fmt, 1, TOSTRING(TEST_TYPE));
  int   id  = -1;
  int   err = nomp_jit(&id, knl, clauses, 2, "a", sizeof(TEST_TYPE), NOMP_PTR,
                       "N", sizeof(int), NOMP_INT);
  nomp_free(&knl);
  return err;
}

#define nomp_api_100_invalid_file                                              \
  TOKEN_PASTE(nomp_api_100_invalid_file, TEST_SUFFIX)
static int nomp_api_100_invalid_file(void) {
  const char *clauses[4] = {"transform", "invalid_file", "tile", 0};
  int         err        = nomp_api_100_aux(clauses);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_PY_CALL_FAILURE);

  char *log = nomp_get_err_str(err);
  int   eq =
      logcmp(log, "\\[Error\\] .*src\\/.*.c:[0-9]* Importing Python module "
                  "\"invalid_file\" failed.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}
#undef nomp_api_100_invalid_file

#define nomp_api_100_invalid_function                                          \
  TOKEN_PASTE(nomp_api_100_invalid_function, TEST_SUFFIX)
static int nomp_api_100_invalid_function(void) {
  const char *clauses[4] = {"transform", "nomp_api_100", "invalid_func", 0};
  int         err        = nomp_api_100_aux(clauses);
  nomp_test_assert(nomp_get_err_no(err) == NOMP_PY_CALL_FAILURE);

  char *log = nomp_get_err_str(err);
  int   eq  = logcmp(
      log, "\\[Error\\] .*src\\/loopy.c:[0-9]* Importing Python function "
              "\"invalid_func\" from module \"nomp_api_100\" failed.");
  nomp_free(&log);
  nomp_test_assert(eq);

  return 0;
}
#undef nomp_api_100_invalid_function
#undef nomp_api_100_aux
