#include "nomp-test.h"

#define nomp_api_240_aux TOKEN_PASTE(nomp_api_240_aux, TEST_SUFFIX)
static int nomp_api_240_aux(const char *knl_fmt, const char **clauses,
                            TEST_TYPE *a, int N) {
  int err = nomp_update(a, 0, N, sizeof(TEST_TYPE), NOMP_TO);
  nomp_test_chk(err);

  int id = -1;
  err = create_knl(&id, knl_fmt, clauses, 1, TOSTRING(TEST_TYPE));
  nomp_test_chk(err);

  nomp_run(id, 2, "a", NOMP_PTR, sizeof(TEST_TYPE), a, "N", NOMP_INT,
           sizeof(int), &N);
  nomp_test_chk(err);

  err = nomp_update(a, 0, N, sizeof(TEST_TYPE), NOMP_FROM);
  nomp_test_chk(err);
  err = nomp_update(a, 0, N, sizeof(TEST_TYPE), NOMP_FREE);
  nomp_test_chk(err);

  return 0;
}

#define nomp_api_240_break TOKEN_PASTE(nomp_api_240_break, TEST_SUFFIX)
static int nomp_api_240_break(int N) {
  nomp_test_assert(N <= 10);

  TEST_TYPE a[10];
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
  const char *clauses[4] = {"transform", "nomp-api-240", "transform", 0};
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
#undef nomp_api_240_aux
