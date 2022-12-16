#include "nomp-test.h"

#define nomp_api_210 TOKEN_PASTE(nomp_api_210, TEST_SUFFIX)
int nomp_api_210() {
  TEST_TYPE a[20] = {0}, b[20] = {1, 2, 3, 4, 5};
  int N = 20;

  int err = nomp_update(a, 0, 20, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);
  err = nomp_update(b, 0, 20, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = 2 * b[i] + 1;                               \n"
      "}                                                      \n";

  const char *clauses[4] = {"transform", "nomp-api-200", "transform", 0};

  char *knl = create_knl(knl_fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  run_kernel(knl, clauses, 3, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "b", NOMP_PTR,
                 sizeof(TEST_TYPE), b, "N", NOMP_INTEGER, sizeof(int), &N);

  err = nomp_update(a, 0, 20, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_test_chk(err);

#if defined(TEST_TOL)
  for (int i = 0; i < N; i++)
    nomp_test_assert(fabs(a[i] - 2 * b[i] - 1) < TEST_TOL);
#else
  for (int i = 0; i < N; i++)
    nomp_test_assert(a[i] == 2 * b[i] + 1);
#endif

  err = nomp_update(a, 0, 20, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);
  err = nomp_update(b, 0, 20, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);

  return 0;
}
#undef nomp_api_210
