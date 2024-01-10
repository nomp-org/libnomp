#include "nomp-test.h"

#define nomp_api_600_aux TOKEN_PASTE(nomp_api_600_aux, TEST_SUFFIX)
static int nomp_api_600_aux(const char *fmt, TEST_TYPE *a, TEST_TYPE *b,
                            int n) {
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_check(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  const char *clauses[4] = {"annotate", "grid_loop", "i", 0};
  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(TEST_TYPE));
  nomp_test_check(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE),
                           NOMP_PTR, "b", sizeof(TEST_TYPE), NOMP_PTR, "N",
                           sizeof(int), NOMP_INT));
  nomp_free(&knl);

  nomp_test_check(nomp_run(id, a, b, &n));

  nomp_test_check(nomp_sync());

  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_check(nomp_update(b, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_600_add TOKEN_PASTE(nomp_api_600_add, TEST_SUFFIX)
static int nomp_api_600_add(unsigned n) {
  nomp_test_assert(n <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE], b[TEST_MAX_SIZE];
  for (unsigned i = 0; i < n; i++)
    a[i] = n - i, b[i] = i;

  const char *knl_fmt =
      "void foo(%s *a, %s *b, int N) {                        \n"
      "  for (int i = 0; i < N; i++)                          \n"
      "    a[i] += b[i];                                      \n"
      "}                                                      \n";
  nomp_api_600_aux(knl_fmt, a, b, n);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(fabs(a[i] - n) < TEST_TOL);
#else
  for (unsigned i = 0; i < n; i++)
    nomp_test_assert(a[i] == (TEST_TYPE)n);
#endif

  return 0;
}
#undef nomp_api_600_add
