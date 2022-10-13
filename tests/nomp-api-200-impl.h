#include "nomp-test.h"

#define nomp_api_200 TOKEN_PASTE(nomp_api_200, TEST_SUFFIX)
int nomp_api_200() {
  const char *knl_fmt =
      "void foo(%s *a, int N) {                                             \n"
      "  for (int i = 0; i < N; i++)                                        \n"
      "    a[i] = i;                                                        \n"
      "}                                                                    \n";

  size_t len = strlen(knl_fmt) + strlen(TOSTRING(TEST_TYPE)) + 1;
  char *knl = calloc(len, sizeof(char));
  snprintf(knl, len, knl_fmt, TOSTRING(TEST_TYPE));

  TEST_TYPE a[10] = {0};
  int N = 10;

  // Calling nomp_jit with invalid functions should return an error.
  static int id = -1;
  const char *annotations[1] = {0},
             *clauses0[3] = {"transform", "invalid-file:invalid_func", 0};
  int err = nomp_jit(&id, knl, annotations, clauses0, 2, "a,N", NOMP_PTR,
                     sizeof(TEST_TYPE), a, NOMP_INTEGER, sizeof(int), &N);
  nomp_assert(nomp_get_log_no(err) == NOMP_USER_CALLBACK_NOT_FOUND);

  const char *clauses1[3] = {"transform", "nomp-api-200:transform", 0};
  err = nomp_jit(&id, knl, annotations, clauses1, 2, "a,N", NOMP_PTR,
                 sizeof(TEST_TYPE), a, NOMP_INTEGER, sizeof(int), &N);
  nomp_chk(err);
  free(knl);

  return 0;
}
#undef nomp_api_200
