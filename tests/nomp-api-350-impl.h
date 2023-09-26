#include "nomp-test.h"

#define nomp_api_350_aux TOKEN_PASTE(nomp_api_350_aux, TEST_SUFFIX)
static int nomp_api_350_aux(const char *fmt, const char **clauses, TEST_TYPE *a,
                            int *b, int n) {
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_check(nomp_update(b, 0, n + 1, sizeof(int), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 2, TOSTRING(TEST_TYPE), TOSTRING(int));
  nomp_test_check(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE),
                           NOMP_PTR, "b", sizeof(int), NOMP_PTR, "N",
                           sizeof(int), NOMP_INT));
  nomp_free(&knl);

  nomp_test_check(nomp_run(id, a, b, &n));

  nomp_test_check(nomp_sync());

  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_check(nomp_update(b, 0, n + 1, sizeof(int), NOMP_FREE));

  return 0;
}

#define nomp_api_350_for_loop_bounds                                           \
  TOKEN_PASTE(nomp_api_350_for_loop_bounds, TEST_SUFFIX)
static int nomp_api_350_for_loop_bounds(unsigned N) {
  nomp_test_assert(N <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE];
  int b[TEST_MAX_SIZE + 1];
  for (unsigned i = 0; i < N; i++)
    a[i] = 0;
  for (unsigned i = 0; i < N + 1; i++)
    b[i] = 2 * i;

  const char *knl_fmt =
      "void foo(%s *a, int *b, int N) {                                \n"
      "  for (int i = 0; i < N; i++) {                                 \n"
      "    int t = 0;                                                  \n"
      "    for (int j = b[i]; j < b[i + 1] + 1; j++) {                 \n"
      "      t += 1;                                                   \n"
      "    }                                                           \n"
      "    a[i] = t;                                                   \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"transform", "nomp_api_215", "tile_outer", 0};
  nomp_api_350_aux(knl_fmt, clauses, a, b, N);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < N; i++)
    nomp_test_assert(fabs(a[i] - 3) < TEST_TOL);
#else
  for (unsigned i = 0; i < N; i++)
    nomp_test_assert(a[i] == 3);
#endif

  return 0;
}
#undef nomp_api_350_for_loop_bounds
#undef nomp_api_350_aux
