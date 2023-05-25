#include "nomp-test.h"

#define nomp_api_243_aux TOKEN_PASTE(nomp_api_243_aux, TEST_SUFFIX)
static int nomp_api_243_aux(const char *fmt, const char **clauses, TEST_TYPE *a,
                            int *b, int n) {
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));
  nomp_test_chk(nomp_update(b, 0, n + 1, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 1, TOSTRING(TEST_TYPE));
  nomp_test_chk(nomp_jit(&id, knl, clauses, 3, "a", sizeof(TEST_TYPE *),
                         NOMP_PTR, "b", sizeof(int *), NOMP_PTR, "N",
                         sizeof(int), NOMP_INT));
  nomp_free(knl);

  nomp_test_chk(nomp_run(id, a, b, &n));

  nomp_test_chk(nomp_sync());

  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_chk(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));
  nomp_test_chk(nomp_update(b, 0, n + 1, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_243_for_loop_bounds                                           \
  TOKEN_PASTE(nomp_api_243_for_loop_bounds, TEST_SUFFIX)
static int nomp_api_243_for_loop_bounds(int N) {
  nomp_test_assert(N <= 10);

  TEST_TYPE a[10];
  int b[11];
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
  const char *clauses[4] = {"transform", "nomp-api-240", "transform", 0};
  nomp_api_243_aux(knl_fmt, clauses, a, b, N);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < N; i++)
    nomp_test_assert(fabs(a[i] - 3) < TEST_TOL);
#else
  for (unsigned i = 0; i < N; i++)
    nomp_test_assert(a[i] == 3);
#endif

  return 0;
}
#undef nomp_api_243_for_loop_bounds
#undef nomp_api_243_aux