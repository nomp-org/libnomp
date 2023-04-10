#include "nomp-test.h"

#define nomp_api_200 TOKEN_PASTE(nomp_api_200, TEST_SUFFIX)
static int nomp_api_200(const char **clauses) {
  const char *knl_fmt =
      "void foo(%s *a, int N) {                                             \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i;                                                        \n"
      "}                                                                    \n";
  int id = -1;
  return create_knl(&id, knl_fmt, clauses, 1, TOSTRING(TEST_TYPE));
}

#define nomp_api_200_err TOKEN_PASTE(nomp_api_200_err, TEST_SUFFIX)
static int nomp_api_200_err(const char **clauses) {
  nomp_test_assert(nomp_get_log_no(nomp_api_200(clauses)) ==
                   NOMP_USER_INPUT_IS_INVALID);
  return 0;
}
#undef nomp_api_200_err
#undef nomp_api_200
