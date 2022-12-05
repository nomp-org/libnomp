#include "nomp-test.h"

#define nomp_api_200 TOKEN_PASTE(nomp_api_200, TEST_SUFFIX)
int nomp_api_200() {
  const char *KNL_FMT =
      "void foo(%s *a, int N) {                                             \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i;                                                        \n"
      "}                                                                    \n";

  const char *TYPE_STR = TOSTRING(TEST_TYPE);
  size_t len = strlen(KNL_FMT) + strlen(TYPE_STR) + 1;
  char *knl = tcalloc(char, len);
  snprintf(knl, len, KNL_FMT, TYPE_STR);

  int N = 10;
  TEST_TYPE a[10] = {0};

  // Calling nomp_jit with invalid functions should return an error.
  const char *clauses0[4] = {"transform", "invalid-file", "invalid_func", 0};
  static int id = -1;
  int err = nomp_jit(&id, knl, clauses0, 2, "a", NOMP_PTR, sizeof(TEST_TYPE),
                     "N", NOMP_INT, sizeof(int));
  nomp_assert(nomp_get_log_no(err) == NOMP_PY_CALL_FAILED);

  // With valid parameters, there shouldn't be any issue.
  const char *clauses1[4] = {"transform", "nomp-api-200", "foo", 0};
  err = nomp_jit(&id, knl, clauses1, 2, "a", NOMP_PTR, sizeof(TEST_TYPE), "N",
                 NOMP_INT, sizeof(int));
  nomp_chk(err);

  tfree(knl);

  return 0;
}
#undef nomp_api_200
