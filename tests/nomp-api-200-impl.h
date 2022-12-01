#include "nomp-test.h"

#define nomp_api_200 TOKEN_PASTE(nomp_api_200, TEST_SUFFIX)
int nomp_api_200() {
  const char *knl_fmt =
      "void foo(%s *a, int N) {                                             \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i;                                                        \n"
      "}                                                                    \n";

  size_t len = strlen(knl_fmt) + strlen(TOSTRING(TEST_TYPE)) + 1;
  char *knl = tcalloc(char, len);
  snprintf(knl, len, knl_fmt, TOSTRING(TEST_TYPE));

  // Calling nomp_jit with invalid functions should return an error.
  static int id = -1;
  const char *clauses0[4] = {"transform", "invalid-file", "invalid_func", 0};
  int err = nomp_jit(&id, knl, clauses0);
  nomp_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILED);

  const char *clauses1[4] = {"transform", "nomp-api-200", "foo", 0};
  err = nomp_jit(&id, knl, clauses1);
  nomp_chk(err);
  tfree(knl);

  return 0;
}
#undef nomp_api_200
