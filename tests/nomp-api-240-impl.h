#include "nomp-test.h"

#define nomp_api_240_aux TOKEN_PASTE(nomp_api_240_aux, TEST_SUFFIX)
static int nomp_api_240_aux(const char *fmt, const char **clauses, TEST_TYPE *a,
                            int n) {
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_TO));

  int id = -1;
  char *knl = generate_knl(fmt, 1, TOSTRING(TEST_TYPE));
  nomp_test_check(nomp_jit(&id, knl, clauses, 2, "a", sizeof(TEST_TYPE),
                           NOMP_PTR, "N", sizeof(int), NOMP_INT));
  nomp_free(&knl);

  nomp_test_check(nomp_run(id, a, &n));

  nomp_test_check(nomp_sync());

  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FROM));
  nomp_test_check(nomp_update(a, 0, n, sizeof(TEST_TYPE), NOMP_FREE));

  return 0;
}

#define nomp_api_240_break TOKEN_PASTE(nomp_api_240_break, TEST_SUFFIX)
static int nomp_api_240_break(unsigned N) {
  nomp_test_assert(N <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE];
  for (unsigned i = 0; i < N; i++)
    a[i] = 0;

  const char *knl_fmt =
      "void foo(%s *a,  int N) {                                       \n"
      "  for (int i = 0; i < N; i++) {                                 \n"
      "    int t = 0;                                                  \n"
      "    for (int j = 0; j < 10; j++) {                              \n"
      "      t += 1;                                                   \n"
      "      if (j == 5)                                               \n"
      "        break;                                                  \n"
      "    }                                                           \n"
      "    a[i] = t;                                                   \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"transform", "nomp_api_215", "tile_outer", 0};
  nomp_api_240_aux(knl_fmt, clauses, a, N);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < N; i++)
    nomp_test_assert(fabs(a[i] - 6) < TEST_TOL);
#else
  for (unsigned i = 0; i < N; i++)
    nomp_test_assert(a[i] == 6);
#endif

  return 0;
}
#undef nomp_api_240_break

#define nomp_api_240_continue TOKEN_PASTE(nomp_api_240_continue, TEST_SUFFIX)
static int nomp_api_240_continue(unsigned N) {
  nomp_test_assert(N <= TEST_MAX_SIZE);

  TEST_TYPE a[TEST_MAX_SIZE];
  for (unsigned i = 0; i < N; i++)
    a[i] = 0;

  const char *knl_fmt =
      "void foo(%s *a,  int N) {                                       \n"
      "  for (int i = 0; i < N; i++) {                                 \n"
      "    int t = 0;                                                  \n"
      "    for (int j = 0; j < 10; j++) {                              \n"
      "      if (j == 5)                                               \n"
      "        continue;                                               \n"
      "      t += 1;                                                   \n"
      "    }                                                           \n"
      "    a[i] = t;                                                   \n"
      "  }                                                             \n"
      "}                                                               \n";
  const char *clauses[4] = {"transform", "nomp_api_215", "tile_outer", 0};
  nomp_api_240_aux(knl_fmt, clauses, a, N);

#if defined(TEST_TOL)
  for (unsigned i = 0; i < N; i++)
    nomp_test_assert(fabs(a[i] - 9) < TEST_TOL);
#else
  for (unsigned i = 0; i < N; i++)
    nomp_test_assert(a[i] == 9);
#endif

  return 0;
}
#undef nomp_api_240_continue
#undef nomp_api_240_aux
