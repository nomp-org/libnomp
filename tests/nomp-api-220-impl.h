#include "nomp-test.h"

#define nomp_api_220_aux TOKEN_PASTE(nomp_api_220_aux, TEST_SUFFIX)
int nomp_api_220_aux(TEST_TYPE *a, TEST_TYPE *b, int N) {
  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] = a[i] + b[i];                                \n"
      "}                                                      \n";
  const char *clauses[4] = {"transform", "nomp-api-200", "transform", 0};

  char *knl = create_knl(knl_fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  return run_kernel(knl, clauses, 3, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "b", NOMP_PTR,
                 sizeof(TEST_TYPE), b, "N", NOMP_INTEGER, sizeof(int), &N);
}

#define nomp_api_220 TOKEN_PASTE(nomp_api_220, TEST_SUFFIX)
int nomp_api_220(int n) {
  nomp_test_assert(n <= 20);
  TEST_TYPE a[20], b[20];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  int err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);

  nomp_api_220_aux(a, b, n);

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_test_chk(err);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == n);
#endif

  err = nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);
  err = nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);

  return 0;
}
#undef nomp_api_220

#undef nomp_api_220_aux
